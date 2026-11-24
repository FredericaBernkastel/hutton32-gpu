//! `gpu-insert`: concurrent find-or-insert into a GPU hash table using only the 32-bit
//! compare-exchange that WGSL offers. This is the primitive the miss path needs (hash-consing new
//! nodes and deduplicating identical misses inside one batch).
//!
//! Protocol (one dispatch, lock-free, no retries across dispatches):
//!   * `claim[slot]`: 0 = empty, `j+1` = claimed in this batch by batch element `j`,
//!     bit 31 set = committed in an earlier batch.
//!   * a thread that finds a slot claimed by element `t` does NOT read the half-written key in the
//!     table; it reads element `t`'s key from the immutable batch input instead, so every
//!     comparison is race-free.
//!   * a second, trivial dispatch marks the batch's claims as committed.
//! The slot index is the node id, so entries never move.

use crate::{gpu::Gpu, opt, opt_or};

const WGSL: &str = r#"
struct P { mask: u32, n: u32, row: u32, universe: u32, seed: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> claim: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_id: array<u32>;

fn rnd(x0: u32) -> u32 {
  var x = x0;
  x ^= x >> 16u; x *= 0x7FEB352Du; x ^= x >> 15u; x *= 0x846CA68Bu; x ^= x >> 16u;
  return x;
}
// the batch input: element j asks for key `key_of(j)`; many elements share a key
fn key_of(j: u32) -> vec4<u32> {
  // p.p0 selects the key universe: 0 and 1 are disjoint, 2 straddles universe 0 and a new one
  var u = rnd(j ^ p.seed) % p.universe;
  if (p.p0 == 1u) { u += p.universe; }
  if (p.p0 == 2u) { u += p.universe / 2u * 3u; if ((u & 1u) == 0u) { u = (u - p.universe / 2u * 3u); } }
  return vec4<u32>(rnd(u * 4u + 1u) & 0x7FFFFFFFu, rnd(u * 4u + 2u), rnd(u * 4u + 3u), u);
}
fn hash4(k: vec4<u32>) -> u32 {
  var h = k.x * 0x9E3779B1u;
  h = (h ^ (h >> 15u)) + k.y * 0x85EBCA77u;
  h = (h ^ (h >> 13u)) + k.z * 0xC2B2AE3Du;
  h = (h ^ (h >> 16u)) + k.w * 0x27D4EB2Fu;
  h = (h ^ (h >> 15u)) * 0x2C1B3C6Du;
  return h ^ (h >> 12u);
}
@compute @workgroup_size(256) fn insert(@builtin(global_invocation_id) id: vec3<u32>) {
  let j = id.x + id.y * p.row;
  if (j >= p.n) { return; }
  let k = key_of(j);
  var slot = hash4(k) & p.mask;
  var res = 0xFFFFFFFFu;
  for (var i = 0u; i < 1024u; i++) {
    var c = atomicLoad(&claim[slot]);
    if (c == 0u) {
      let r = atomicCompareExchangeWeak(&claim[slot], 0u, j + 1u);
      if (r.exchanged) {
        let b = slot * 4u;
        keys[b] = k.x; keys[b + 1u] = k.y; keys[b + 2u] = k.z; keys[b + 3u] = k.w;
        res = slot;
        break;
      }
      c = r.old_value;
      if (c == 0u) { continue; }  // spurious failure: try the same slot again
    }
    var other: vec4<u32>;
    if ((c & 0x80000000u) != 0u) {
      let b = slot * 4u;
      other = vec4<u32>(keys[b], keys[b + 1u], keys[b + 2u], keys[b + 3u]);  // committed: immutable
    } else {
      other = key_of(c - 1u);  // pending: read the owner's key from the batch input
    }
    if (all(other == k)) { res = slot; break; }
    slot = (slot + 1u) & p.mask;
  }
  out_id[j] = res;
}
@compute @workgroup_size(256) fn commit(@builtin(global_invocation_id) id: vec3<u32>) {
  let j = id.x + id.y * p.row;
  if (j >= p.n) { return; }
  let s = out_id[j];
  if (s != 0xFFFFFFFFu) {
    if (atomicLoad(&claim[s]) == j + 1u) { atomicStore(&claim[s], 0x80000000u); }
  }
}
"#;

fn rnd(x0: u32) -> u32 {
  let mut x = x0;
  x ^= x >> 16;
  x = x.wrapping_mul(0x7FEB352D);
  x ^= x >> 15;
  x = x.wrapping_mul(0x846CA68B);
  x ^= x >> 16;
  x
}
fn key_of(j: u32, seed: u32, universe: u32, off: u32) -> [u32; 4] {
  let mut u = rnd(j ^ seed) % universe;
  if off == 1 {
    u += universe;
  }
  if off == 2 {
    u += universe / 2 * 3;
    if u & 1 == 0 {
      u -= universe / 2 * 3;
    }
  }
  [rnd(u.wrapping_mul(4).wrapping_add(1)) & 0x7FFF_FFFF, rnd(u.wrapping_mul(4).wrapping_add(2)), rnd(u.wrapping_mul(4).wrapping_add(3)), u]
}

pub fn run(args: &[String]) {
  let gpu = Gpu::new(opt(args, "--backend").as_deref());
  println!("GPU: {}", gpu.describe());
  let log2_slots: u32 = opt_or(args, "--log2-slots", 25);
  let n: u32 = opt_or(args, "--n", 1 << 24);
  let universe: u32 = opt_or(args, "--universe", 1 << 22);
  let slots = 1u64 << log2_slots;
  let mask = (slots - 1) as u32;
  let claim = gpu.storage("claim", slots * 4);
  let keys = gpu.storage("keys", slots * 16);
  let insert = gpu.pipeline("insert", WGSL, "insert");
  let commit = gpu.pipeline("commit", WGSL, "commit");
  let row = 1u32 << 22;
  println!(
    "table 2^{log2_slots} slots (claim {} MiB + keys {} MiB); batch = {n} requests drawn from {universe} keys",
    slots * 4 >> 20,
    slots * 16 >> 20
  );
  // batch plan: (label, seed, key-universe offset). Batches A and B insert disjoint key sets,
  // C repeats A's requests exactly (pure find of committed keys), D mixes old and new keys.
  let plan: [(&str, u32, u32); 4] = [
    ("A: all keys new          ", 0xC0FFEE, 0),
    ("B: all keys new          ", 0xBEEF01, 1),
    ("C: all keys already there", 0xC0FFEE, 0),
    ("D: half new, half old    ", 0x5EED77, 2),
  ];
  let outs: Vec<wgpu::Buffer> = plan.iter().map(|_| gpu.storage("out", n as u64 * 4)).collect();
  let mut times = Vec::new();
  for (b, &(_, seed, off)) in plan.iter().enumerate() {
    let uni = gpu.device.create_buffer(&wgpu::BufferDescriptor {
      label: None,
      size: 32,
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&uni, 0, bytemuck::cast_slice(&[mask, n, row, universe, seed, off, 0, 0]));
    let bg_i = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: None,
      layout: &insert.get_bind_group_layout(0),
      entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: uni.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: claim.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: keys.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: outs[b].as_entire_binding() },
      ],
    });
    let bg_c = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: None,
      layout: &commit.get_bind_group_layout(0),
      entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: uni.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: claim.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: outs[b].as_entire_binding() },
      ],
    });
    let (gx, gy) = ((row / 256).min(n.div_ceil(256)), n.div_ceil(row));
    let t_insert = gpu.timed(|| {
      let mut enc = gpu.device.create_command_encoder(&Default::default());
      {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&insert);
        pass.set_bind_group(0, &bg_i, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
      }
      gpu.queue.submit(Some(enc.finish()));
    });
    let t_commit = gpu.timed(|| {
      let mut enc = gpu.device.create_command_encoder(&Default::default());
      {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&commit);
        pass.set_bind_group(0, &bg_c, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
      }
      gpu.queue.submit(Some(enc.finish()));
    });
    times.push((t_insert, t_commit));
  }
  // ---- verify everything on the CPU ----
  let tab = gpu.read_u32(&keys, 0, slots * 4);
  let cl = gpu.read_u32(&claim, 0, slots);
  let mut global: std::collections::HashMap<[u32; 4], u32> = Default::default();
  for (b, &(label, seed, off)) in plan.iter().enumerate() {
    let ids = gpu.read_u32(&outs[b], 0, n as u64);
    let before = global.len();
    let (mut unresolved, mut wrong_key, mut dup, mut uncommitted) = (0u64, 0u64, 0u64, 0u64);
    let mut in_batch: std::collections::HashSet<[u32; 4]> = Default::default();
    for j in 0..n {
      let k = key_of(j, seed, universe, off);
      let s = ids[j as usize];
      if s == u32::MAX {
        unresolved += 1;
        continue;
      }
      if tab[s as usize * 4..s as usize * 4 + 4] != k {
        wrong_key += 1;
      }
      if cl[s as usize] != 0x8000_0000 {
        uncommitted += 1;
      }
      if *global.entry(k).or_insert(s) != s {
        dup += 1;
      }
      in_batch.insert(k);
    }
    let (ti, tc) = times[b];
    println!(
      "  {label}: {:.3e} find-or-insert/s (insert {:.1} ms + commit {:.1} ms) | {} distinct in batch, {} new | unresolved {unresolved}, wrong key {wrong_key}, duplicate ids {dup}, uncommitted {uncommitted}",
      n as f64 / (ti + tc),
      ti * 1e3,
      tc * 1e3,
      in_batch.len(),
      global.len() - before
    );
  }
  let used = cl.iter().filter(|&&c| c != 0).count();
  println!("  slots used {used}, distinct keys overall {} -> {}", global.len(), if used == global.len() { "consistent" } else { "INCONSISTENT" });
}
