//! Memo tables in VRAM (Stage 2): open-addressed tables of 32-byte entries with linear probing,
//! a node's ID being its slot index; every level shares one table, the level being part of the key.
//! Base tiles live in a tile table whose slots index the tile store, the rule's base tiles as
//! bit-planes ([`crate::tile::TileGeom`]).
//!
//! Node entry, eight words: `[k0, k1, k2, k3, result, claim, level | tag << 8, population]`.
//! Tile slot, four words: `[claim, hash, population, unused]`; the tile's words at `slot * W` of the
//! store.
//!
//! Every insert, the host's included, goes through one batch find-or-insert with the claim protocol
//! of the paper (`cmd_insert.rs`): the claim word is 0 (empty), `j + 1` (claimed in this batch by
//! element `j`) or has bit 31 set (committed). A thread that finds a slot claimed by element `t`
//! compares against `t`'s key in the immutable batch input, never the table; committed slots are
//! immutable and compared in place; a second dispatch commits the batch. One dispatch resolves a
//! whole batch without locks.

use rustc_hash::FxHashMap;

use super::context::GpuContext;
use crate::{
  cpu::Engine,
  tile::{hash4, tile_hash, NONE},
  Error, Result,
};

/// Words per node entry and per tile slot.
pub const ENTRY: u64 = 8;
pub const TILE_SLOT: u64 = 4;
/// Committed bit of a claim word.
pub const COMMITTED: u32 = 0x8000_0000;

/// Slot of a key: `hash4(key)` mixed with the level.
#[inline]
pub fn node_home(key: [u32; 4], level: u32, mask: u32) -> u32 {
  (hash4(key) ^ level.wrapping_mul(0x9E37_79B9)) & mask
}

/// WGSL of the node hash, shared by every kernel that touches the node table.
pub fn common_wgsl() -> String {
  "fn hash4(k: vec4<u32>) -> u32 {
  var h = k.x * 0x9E3779B1u;
  h = (h ^ (h >> 15u)) + k.y * 0x85EBCA77u;
  h = (h ^ (h >> 13u)) + k.z * 0xC2B2AE3Du;
  h = (h ^ (h >> 16u)) + k.w * 0x27D4EB2Fu;
  h = (h ^ (h >> 15u)) * 0x2C1B3C6Du;
  return h ^ (h >> 12u);
}
fn home(k: vec4<u32>, level: u32, mask: u32) -> u32 { return (hash4(k) ^ (level * 0x9E3779B9u)) & mask; }
"
  .to_string()
}

/// WGSL of the tile hash (`crate::tile::tile_hash`) over `n` words read by `tw_in(i)`.
pub fn tile_hash_wgsl() -> String {
  "fn tile_hash(base: u32, n: u32) -> u32 {
  var h = 0x9E3779B9u;
  for (var i = 0u; i < n; i++) {
    h ^= tw_in(base + i);
    h = h * 0x85EBCA6Bu;
    h ^= h >> 13u;
  }
  h ^= h >> 16u;
  h = h * 0xC2B2AE35u;
  return h ^ (h >> 16u);
}
"
  .to_string()
}

/// The first binding of the node table's shards in every kernel.
pub const NODE_BINDING: u32 = 16;

/// Tables beyond one binding (Stage 5): the node table lives in `shards` buffers of `2^shard_log2`
/// entries, a slot's shard being its top bits. A kernel written against one array `nodes` is
/// rewritten to reach the shards through accessors: `atomicLoad(&nodes[i])` and plain reads become
/// `n_ld(i)`, `atomicStore(&nodes[i], v)` and plain writes `n_st(i, v)`,
/// `atomicCompareExchangeWeak(&nodes[i], a, b)` becomes `n_cas(i, a, b)`; the declaration of
/// `nodes` is replaced by the shards' at bindings `NODE_BINDING..`.
pub fn shard_nodes(src: &str, shards: u32, shard_log2: u32) -> String {
  let atomic = src.lines().any(|l| l.contains(" nodes:") && l.contains("atomic"));
  let write = src.lines().any(|l| l.contains(" nodes:") && l.contains("read_write"));
  let mut out = String::with_capacity(src.len() + 1024);
  for line in src.lines() {
    if line.contains("var<storage") && line.contains(" nodes:") {
      continue;
    }
    out.push_str(line);
    out.push('\n');
  }
  // rewrite the accesses
  let mut s = out;
  let mut res = String::with_capacity(s.len());
  loop {
    let Some(p) = s.find("nodes[") else {
      res.push_str(&s);
      break;
    };
    // the index expression up to the matching bracket
    let open = p + "nodes".len();
    let mut depth = 0i32;
    let mut close = open;
    for (i, ch) in s[open..].char_indices() {
      match ch {
        '[' => depth += 1,
        ']' => {
          depth -= 1;
          if depth == 0 {
            close = open + i;
            break;
          }
        }
        _ => {}
      }
    }
    let idx = s[open + 1..close].to_string();
    let before = &s[..p];
    let after = &s[close + 1..];
    let wrap = |pre: &str| before.ends_with(pre);
    if wrap("atomicLoad(&") {
      res.push_str(&before[..before.len() - "atomicLoad(&".len()]);
      res.push_str(&format!("n_ld({idx}"));
      s = after.to_string();
    } else if wrap("atomicStore(&") {
      res.push_str(&before[..before.len() - "atomicStore(&".len()]);
      res.push_str(&format!("n_st({idx}"));
      s = after.to_string();
    } else if wrap("atomicCompareExchangeWeak(&") {
      res.push_str(&before[..before.len() - "atomicCompareExchangeWeak(&".len()]);
      res.push_str(&format!("n_cas({idx}"));
      s = after.to_string();
    } else if after.trim_start().starts_with('=') && !after.trim_start().starts_with("==") {
      // a plain write: nodes[i] = v;
      let rhs_start = after.find('=').unwrap() + 1;
      let end = after.find(';').unwrap();
      res.push_str(before);
      res.push_str(&format!("n_st({idx}, {})", after[rhs_start..end].trim()));
      s = after[end..].to_string();
    } else {
      res.push_str(before);
      res.push_str(&format!("n_ld({idx})"));
      s = after.to_string();
    }
  }
  // the shards and their accessors
  let words_log2 = shard_log2 + 3;
  let mask = (1u64 << words_log2) - 1;
  let mut d = String::new();
  for j in 0..shards {
    if atomic {
      d += &format!("@group(0) @binding({}) var<storage, read_write> nodes{j}: array<atomic<u32>>;\n", NODE_BINDING + j);
    } else if write {
      d += &format!("@group(0) @binding({}) var<storage, read_write> nodes{j}: array<u32>;\n", NODE_BINDING + j);
    } else {
      d += &format!("@group(0) @binding({}) var<storage, read> nodes{j}: array<u32>;\n", NODE_BINDING + j);
    }
  }
  let arm = |body: &dyn Fn(u32) -> String| -> String {
    if shards == 1 {
      return format!("  let l = i;\n  {}\n", body(0));
    }
    let mut a = format!("  let l = i & {mask}u;\n  switch (i >> {words_log2}u) {{\n");
    for j in 0..shards {
      if j + 1 < shards {
        a += &format!("    case {j}u: {{ {} }}\n", body(j));
      } else {
        a += &format!("    default: {{ {} }}\n", body(j));
      }
    }
    a + "  }\n"
  };
  if atomic {
    d += "struct CasR { old_value: u32, exchanged: bool };\n";
    d += &format!("fn n_ld(i: u32) -> u32 {{\n{}}}\n", arm(&|j| format!("return atomicLoad(&nodes{j}[l]);")));
    d += &format!("fn n_st(i: u32, v: u32) {{\n{}}}\n", arm(&|j| format!("atomicStore(&nodes{j}[l], v); return;")));
    d += &format!(
      "fn n_cas(i: u32, a: u32, b: u32) -> CasR {{\n{}}}\n",
      arm(&|j| format!("let r = atomicCompareExchangeWeak(&nodes{j}[l], a, b); return CasR(r.old_value, r.exchanged);"))
    );
  } else {
    d += &format!("fn n_ld(i: u32) -> u32 {{\n{}}}\n", arm(&|j| format!("return nodes{j}[l];")));
    if write {
      d += &format!("fn n_st(i: u32, v: u32) {{\n{}}}\n", arm(&|j| format!("nodes{j}[l] = v; return;")));
    }
  }
  // declarations first: WGSL resolves names module-wide, but keep them readable at the top
  format!("{d}{res}")
}

/// Find in the node table (read-only probe): the slot of `(level, key)` or `0xFFFFFFFF`.
pub fn find_wgsl(nodes: &str) -> String {
  format!(
    "fn find(k: vec4<u32>, level: u32, mask: u32) -> u32 {{
  var slot = home(k, level, mask);
  for (var i = 0u; i <= mask; i++) {{
    let b = slot * 8u;
    let c = {nodes}[b + 5u];
    if (c == 0u) {{ return 0xFFFFFFFFu; }}
    if ({nodes}[b] == k.x && {nodes}[b + 1u] == k.y && {nodes}[b + 2u] == k.z && {nodes}[b + 3u] == k.w && ({nodes}[b + 6u] & 0xFFu) == level) {{ return slot; }}
    slot = (slot + 1u) & mask;
  }}
  return 0xFFFFFFFFu;
}}
"
  )
}

/// Batch find-or-insert of nodes: claim, then commit.
const CLAIM_NODES: &str = "struct B { n: u32, level: u32, mask: u32, pad: u32 };
@group(0) @binding(0) var<uniform> bp: B;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read> pops: array<u32>;
@group(0) @binding(3) var<storage, read_write> nodes: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> ids: array<u32>;
fn key_of(j: u32) -> vec4<u32> { return vec4<u32>(keys[j * 4u], keys[j * 4u + 1u], keys[j * 4u + 2u], keys[j * 4u + 3u]); }
@compute @workgroup_size(64) fn claim(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x + gid.y * 65535u * 64u;
  if (j >= bp.n) { return; }
  let k = key_of(j);
  var slot = home(k, bp.level, bp.mask);
  for (var i = 0u; i <= bp.mask; i++) {
    let b = slot * 8u;
    var c = atomicLoad(&nodes[b + 5u]);
    if (c == 0u) {
      let r = atomicCompareExchangeWeak(&nodes[b + 5u], 0u, j + 1u);
      if (r.exchanged) {
        atomicStore(&nodes[b], k.x);
        atomicStore(&nodes[b + 1u], k.y);
        atomicStore(&nodes[b + 2u], k.z);
        atomicStore(&nodes[b + 3u], k.w);
        atomicStore(&nodes[b + 4u], 0xFFFFFFFFu);
        atomicStore(&nodes[b + 6u], bp.level);
        atomicStore(&nodes[b + 7u], pops[j]);
        ids[j] = slot | 0x80000000u;
        return;
      }
      c = r.old_value;
    }
    if ((c & 0x80000000u) != 0u) {
      if (atomicLoad(&nodes[b]) == k.x && atomicLoad(&nodes[b + 1u]) == k.y && atomicLoad(&nodes[b + 2u]) == k.z
          && atomicLoad(&nodes[b + 3u]) == k.w && (atomicLoad(&nodes[b + 6u]) & 0xFFu) == bp.level) {
        ids[j] = slot;
        return;
      }
    } else if (c != 0u) {
      // claimed in this batch by element c - 1: compare against its key in the batch input
      if (all(key_of(c - 1u) == k)) {
        ids[j] = slot;
        return;
      }
    }
    slot = (slot + 1u) & bp.mask;
  }
  ids[j] = 0xFFFFFFFEu;
}
@compute @workgroup_size(64) fn commit(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x + gid.y * 65535u * 64u;
  if (j >= bp.n) { return; }
  let id = ids[j];
  if (id != 0xFFFFFFFEu && (id & 0x80000000u) != 0u) {
    let slot = id & 0x7FFFFFFFu;
    atomicStore(&nodes[slot * 8u + 5u], 0x80000000u);
    ids[j] = slot;
  }
}
";

/// Batch find-or-insert of tiles by content: `w` words per tile, `planes` planes of `w / planes`
/// words each (the population is counted over the union of the planes).
fn claim_tiles(w: u32, planes: u32) -> String {
  let wpp = w / planes;
  format!(
    "struct B {{ n: u32, w: u32, mask: u32, pad: u32 }};
@group(0) @binding(0) var<uniform> bp: B;
@group(0) @binding(1) var<storage, read> words: array<u32>;
@group(0) @binding(2) var<storage, read_write> slots: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> store: array<u32>;
@group(0) @binding(4) var<storage, read_write> ids: array<u32>;
fn tw_in(i: u32) -> u32 {{ return words[i]; }}
{th}
fn tile_pop(j: u32) -> u32 {{
  var pop = 0u;
  for (var q = 0u; q < {wpp}u; q++) {{
    var v = 0u;
    for (var k = 0u; k < {planes}u; k++) {{ v |= words[j * {w}u + k * {wpp}u + q]; }}
    pop += countOneBits(v);
  }}
  return pop;
}}
fn equal_batch(a: u32, b: u32) -> bool {{
  for (var i = 0u; i < {w}u; i++) {{ if (words[a * {w}u + i] != words[b * {w}u + i]) {{ return false; }} }}
  return true;
}}
fn equal_store(slot: u32, j: u32) -> bool {{
  for (var i = 0u; i < {w}u; i++) {{ if (store[slot * {w}u + i] != words[j * {w}u + i]) {{ return false; }} }}
  return true;
}}
@compute @workgroup_size(64) fn claim(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let j = gid.x + gid.y * 65535u * 64u;
  if (j >= bp.n) {{ return; }}
  let h = tile_hash(j * {w}u, {w}u);
  var slot = h & bp.mask;
  for (var i = 0u; i <= bp.mask; i++) {{
    let b = slot * 4u;
    var c = atomicLoad(&slots[b]);
    if (c == 0u) {{
      let r = atomicCompareExchangeWeak(&slots[b], 0u, j + 1u);
      if (r.exchanged) {{
        atomicStore(&slots[b + 1u], h);
        for (var q = 0u; q < {w}u; q++) {{ store[slot * {w}u + q] = words[j * {w}u + q]; }}
        atomicStore(&slots[b + 2u], tile_pop(j));
        ids[j] = slot | 0x80000000u;
        return;
      }}
      c = r.old_value;
    }}
    if ((c & 0x80000000u) != 0u) {{
      if (atomicLoad(&slots[b + 1u]) == h && equal_store(slot, j)) {{ ids[j] = slot; return; }}
    }} else if (c != 0u) {{
      if (equal_batch(c - 1u, j)) {{ ids[j] = slot; return; }}
    }}
    slot = (slot + 1u) & bp.mask;
  }}
  ids[j] = 0xFFFFFFFEu;
}}
@compute @workgroup_size(64) fn commit(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let j = gid.x + gid.y * 65535u * 64u;
  if (j >= bp.n) {{ return; }}
  let id = ids[j];
  if (id != 0xFFFFFFFEu && (id & 0x80000000u) != 0u) {{
    let slot = id & 0x7FFFFFFFu;
    atomicStore(&slots[slot * 4u], 0x80000000u);
    ids[j] = slot;
  }}
}}
",
    th = tile_hash_wgsl()
  )
}

/// Set results: pairs `(slot, result)`.
const SET_RESULTS: &str = "struct B { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> bp: B;
@group(0) @binding(1) var<storage, read> pairs: array<u32>;
@group(0) @binding(2) var<storage, read_write> nodes: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x + gid.y * 65535u * 64u;
  if (j >= bp.n) { return; }
  nodes[pairs[j * 2u] * 8u + 4u] = pairs[j * 2u + 1u];
}
";

/// The tables, their kernels, and the translation between host and GPU IDs.
pub struct Memo {
  pub a: usize,
  /// words per tile
  pub w: usize,
  pub node_log2: u32,
  pub tile_log2: u32,
  /// the node table in shards of `2^shard_log2` entries, a slot's shard its top bits
  pub nodes: Vec<wgpu::Buffer>,
  pub shard_log2: u32,
  pub tile_slots: wgpu::Buffer,
  pub store: wgpu::Buffer,
  claim_nodes: wgpu::ComputePipeline,
  commit_nodes: wgpu::ComputePipeline,
  claim_tiles: wgpu::ComputePipeline,
  commit_tiles: wgpu::ComputePipeline,
  set_results: wgpu::ComputePipeline,
  /// host ID to GPU ID per level (level `a` = tiles); GPU to host likewise
  pub h2g: Vec<FxHashMap<u32, u32>>,
  pub g2h: Vec<FxHashMap<u32, u32>>,
  /// entries inserted (an upper bound: duplicates found count once per batch)
  pub node_count: u64,
  pub tile_count: u64,
  /// the empty node's GPU ID per level (`a` = the empty tile)
  pub empty: Vec<u32>,
}

/// Dispatch size for `n` invocations of 64-wide workgroups: x up to 65535 groups, then y.
pub fn grid64(n: u64) -> (u32, u32) {
  let groups = n.div_ceil(64).max(1);
  if groups <= 65535 {
    (groups as u32, 1)
  } else {
    (65535, groups.div_ceil(65535) as u32)
  }
}

impl Memo {
  /// Tables of `2^node_log2` node slots and `2^tile_log2` tile slots, the node table in as many
  /// shards as one binding requires.
  pub fn new(ctx: &GpuContext, a: usize, planes: u32, node_log2: u32, tile_log2: u32) -> Result<Memo> {
    let per_binding = 63 - (ctx.max_binding() / (ENTRY * 4)).leading_zeros();
    Memo::with_shards(ctx, a, planes, node_log2, tile_log2, node_log2.min(per_binding))
  }

  /// The same with shards of `2^shard_log2` entries (at most 8 shards).
  pub fn with_shards(ctx: &GpuContext, a: usize, planes: u32, node_log2: u32, tile_log2: u32, shard_log2: u32) -> Result<Memo> {
    let geom = crate::tile::TileGeom::new(a as u32, planes);
    let w = geom.words();
    let shard_log2 = shard_log2.min(node_log2);
    let shards = 1u32 << (node_log2 - shard_log2);
    let sbytes = ((w as u64) << tile_log2) * 4;
    if (ENTRY << shard_log2) * 4 > ctx.max_binding() || sbytes > ctx.max_binding() || shards > 8 {
      return Err(Error::Gpu(format!("memo tables of 2^{node_log2} entries in {shards} shards exceed the bindings")));
    }
    let nodes: Vec<wgpu::Buffer> = (0..shards).map(|_| ctx.storage("nodes", (ENTRY << shard_log2) * 4)).collect();
    let tile_slots = ctx.storage("tile slots", (TILE_SLOT << tile_log2) * 4);
    let store = ctx.storage("tile store", sbytes);
    let cn = shard_nodes(&format!("{}{}", common_wgsl(), CLAIM_NODES), shards, shard_log2);
    let claim_nodes = ctx.pipeline("claim nodes", &cn, "claim", true)?;
    let commit_nodes = ctx.pipeline("commit nodes", &cn, "commit", true)?;
    let ct = claim_tiles(w as u32, planes);
    let claim_tiles = ctx.pipeline("claim tiles", &ct, "claim", true)?;
    let commit_tiles = ctx.pipeline("commit tiles", &ct, "commit", true)?;
    let set_results = ctx.pipeline("set results", &shard_nodes(SET_RESULTS, shards, shard_log2), "main", true)?;
    Ok(Memo {
      a,
      w,
      node_log2,
      tile_log2,
      nodes,
      shard_log2,
      tile_slots,
      store,
      claim_nodes,
      commit_nodes,
      claim_tiles,
      commit_tiles,
      set_results,
      h2g: Vec::new(),
      g2h: Vec::new(),
      node_count: 0,
      tile_count: 0,
      empty: Vec::new(),
    })
  }

  /// The node table's shards as bind-group entries at `NODE_BINDING..`.
  pub fn node_entries(&self) -> Vec<(u32, &wgpu::Buffer)> {
    self.nodes.iter().enumerate().map(|(j, b)| (NODE_BINDING + j as u32, b)).collect()
  }

  /// Words of the node table from slot `first` on (`n` slots), across shards.
  pub fn read_slots(&self, ctx: &GpuContext, first: u64, n: u64) -> Vec<u32> {
    let mut out = Vec::with_capacity((n * ENTRY) as usize);
    let per = 1u64 << self.shard_log2;
    let mut s = first;
    let end = (first + n).min(1u64 << self.node_log2);
    while s < end {
      let shard = (s / per) as usize;
      let local = s % per;
      let take = (per - local).min(end - s);
      out.extend(ctx.read_u32(&self.nodes[shard], local * ENTRY, take * ENTRY));
      s += take;
    }
    out
  }

  pub fn bytes(&self) -> (u64, u64) {
    ((ENTRY << self.node_log2) * 4, ((TILE_SLOT << self.tile_log2) + ((self.w as u64) << self.tile_log2)) * 4)
  }

  fn level_maps(&mut self, l: usize) {
    while self.h2g.len() <= l {
      self.h2g.push(FxHashMap::default());
      self.g2h.push(FxHashMap::default());
    }
  }

  /// Batch find-or-insert of node keys at `level`; returns the GPU IDs.
  pub fn insert_nodes(&mut self, ctx: &GpuContext, level: u32, keys: &[[u32; 4]], pops: &[u32]) -> Result<Vec<u32>> {
    if keys.is_empty() {
      return Ok(Vec::new());
    }
    let n = keys.len() as u64;
    if (self.node_count + n) * 10 > 9 << self.node_log2 {
      return Err(Error::Gpu("the memo table is full".into()));
    }
    let flat: Vec<u32> = keys.iter().flatten().copied().collect();
    let kb = ctx.storage_init("batch keys", &flat);
    let pb = ctx.storage_init("batch pops", pops);
    let ib = ctx.storage("batch ids", n * 4);
    let uni = ctx.uniform("batch", &[n as u32, level, (1u32 << self.node_log2) - 1, 0]);
    let mut e1: Vec<(u32, &wgpu::Buffer)> = vec![(0, &uni), (1, &kb), (2, &pb), (4, &ib)];
    e1.extend(self.node_entries());
    let mut e2: Vec<(u32, &wgpu::Buffer)> = vec![(0, &uni), (4, &ib)];
    e2.extend(self.node_entries());
    let bg = ctx.bind_at(&self.claim_nodes, &e1);
    let bg2 = ctx.bind_at(&self.commit_nodes, &e2);
    let (gx, gy) = grid64(n);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
      let mut pass = enc.begin_compute_pass(&Default::default());
      pass.set_pipeline(&self.claim_nodes);
      pass.set_bind_group(0, &bg, &[]);
      pass.dispatch_workgroups(gx, gy, 1);
    }
    {
      let mut pass = enc.begin_compute_pass(&Default::default());
      pass.set_pipeline(&self.commit_nodes);
      pass.set_bind_group(0, &bg2, &[]);
      pass.dispatch_workgroups(gx, gy, 1);
    }
    ctx.queue.submit(Some(enc.finish()));
    let ids = ctx.read_u32(&ib, 0, n);
    if ids.iter().any(|&i| i == 0xFFFF_FFFE) {
      return Err(Error::Gpu("the memo table is full".into()));
    }
    self.node_count += n;
    Ok(ids)
  }

  /// Batch find-or-insert of tiles (`w` words each); returns the GPU IDs.
  pub fn insert_tiles(&mut self, ctx: &GpuContext, words: &[u32]) -> Result<Vec<u32>> {
    let n = (words.len() / self.w) as u64;
    if n == 0 {
      return Ok(Vec::new());
    }
    if (self.tile_count + n) * 10 > 9 << self.tile_log2 {
      return Err(Error::Gpu("the tile store is full".into()));
    }
    let wb = ctx.storage_init("batch tiles", words);
    let ib = ctx.storage("batch ids", n * 4);
    let uni = ctx.uniform("batch", &[n as u32, self.w as u32, (1u32 << self.tile_log2) - 1, 0]);
    let bg = ctx.bind(&self.claim_tiles, &[&uni, &wb, &self.tile_slots, &self.store, &ib]);
    let bg2 = ctx.bind_at(&self.commit_tiles, &[(0, &uni), (2, &self.tile_slots), (4, &ib)]);
    let (gx, gy) = grid64(n);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
      let mut pass = enc.begin_compute_pass(&Default::default());
      pass.set_pipeline(&self.claim_tiles);
      pass.set_bind_group(0, &bg, &[]);
      pass.dispatch_workgroups(gx, gy, 1);
    }
    {
      let mut pass = enc.begin_compute_pass(&Default::default());
      pass.set_pipeline(&self.commit_tiles);
      pass.set_bind_group(0, &bg2, &[]);
      pass.dispatch_workgroups(gx, gy, 1);
    }
    ctx.queue.submit(Some(enc.finish()));
    let ids = ctx.read_u32(&ib, 0, n);
    if ids.iter().any(|&i| i == 0xFFFF_FFFE) {
      return Err(Error::Gpu("the tile store is full".into()));
    }
    self.tile_count += n;
    Ok(ids)
  }

  /// Write results: `(slot, result)` pairs.
  pub fn set_results(&self, ctx: &GpuContext, pairs: &[(u32, u32)]) {
    if pairs.is_empty() {
      return;
    }
    let flat: Vec<u32> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    let pb = ctx.storage_init("result pairs", &flat);
    let uni = ctx.uniform("results", &[pairs.len() as u32, 0, 0, 0]);
    let mut e: Vec<(u32, &wgpu::Buffer)> = vec![(0, &uni), (1, &pb)];
    e.extend(self.node_entries());
    let bg = ctx.bind_at(&self.set_results, &e);
    let (gx, gy) = grid64(pairs.len() as u64);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
      let mut pass = enc.begin_compute_pass(&Default::default());
      pass.set_pipeline(&self.set_results);
      pass.set_bind_group(0, &bg, &[]);
      pass.dispatch_workgroups(gx, gy, 1);
    }
    ctx.queue.submit(Some(enc.finish()));
  }

  /// Upload host nodes (and everything below them) that the GPU does not have yet; the GPU IDs
  /// of `ids` (level `l`) come back. Level by level, one batch each.
  pub fn upload(&mut self, ctx: &GpuContext, e: &Engine, l: usize, ids: &[u32]) -> Result<Vec<u32>> {
    let a = self.a;
    // the missing nodes per level, top-down
    let mut need: Vec<Vec<u32>> = vec![Vec::new(); l + 1];
    let mut seen: Vec<rustc_hash::FxHashSet<u32>> = vec![Default::default(); l + 1];
    self.level_maps(l);
    for &id in ids {
      if !self.h2g[l].contains_key(&id) && seen[l].insert(id) {
        need[l].push(id);
      }
    }
    for lv in (a + 1..=l).rev() {
      let list = std::mem::take(&mut need[lv]);
      for &id in &list {
        for c in e.ch(lv, id) {
          if !self.h2g[lv - 1].contains_key(&c) && seen[lv - 1].insert(c) {
            need[lv - 1].push(c);
          }
        }
      }
      need[lv] = list;
    }
    // tiles
    if !need[a].is_empty() {
      let mut words = Vec::with_capacity(need[a].len() * self.w);
      for &t in &need[a] {
        words.extend_from_slice(e.tile_words(t));
      }
      let g = self.insert_tiles(ctx, &words)?;
      for (&h, &gid) in need[a].iter().zip(&g) {
        self.h2g[a].insert(h, gid);
        self.g2h[a].insert(gid, h);
      }
    }
    for lv in a + 1..=l {
      if need[lv].is_empty() {
        continue;
      }
      let keys: Vec<[u32; 4]> = need[lv].iter().map(|&id| e.ch(lv, id).map(|c| self.h2g[lv - 1][&c])).collect();
      let pops: Vec<u32> = need[lv].iter().map(|&id| e.population(lv, id).min(u32::MAX as u64) as u32).collect();
      let g = self.insert_nodes(ctx, lv as u32, &keys, &pops)?;
      for (&h, &gid) in need[lv].iter().zip(&g) {
        self.h2g[lv].insert(h, gid);
        self.g2h[lv].insert(gid, h);
      }
    }
    Ok(ids.iter().map(|id| self.h2g[l][id]).collect())
  }

  /// Upload the results of host nodes of level `l` (the result nodes too).
  pub fn upload_results(&mut self, ctx: &GpuContext, e: &Engine, l: usize, ids: &[u32]) -> Result<()> {
    let with: Vec<u32> = ids.iter().copied().filter(|&id| e.result(l, id) != NONE).collect();
    let res: Vec<u32> = with.iter().map(|&id| e.result(l, id)).collect();
    let g = self.upload(ctx, e, l, &with)?;
    let gr = self.upload(ctx, e, l - 1, &res)?;
    let pairs: Vec<(u32, u32)> = g.into_iter().zip(gr).collect();
    self.set_results(ctx, &pairs);
    Ok(())
  }

  /// Host IDs of GPU IDs of level `l`, reading unknown entries from the tables and interning them
  /// on the host (bottom-up).
  pub fn download(&mut self, ctx: &GpuContext, e: &mut Engine, l: usize, ids: &[u32]) -> Vec<u32> {
    let a = self.a;
    self.level_maps(l);
    // unknown IDs per level, top-down, with their entries
    let mut unknown: Vec<Vec<u32>> = vec![Vec::new(); l + 1];
    let mut keys: Vec<FxHashMap<u32, [u32; 4]>> = vec![FxHashMap::default(); l + 1];
    for &id in ids {
      if !self.g2h[l].contains_key(&id) && !unknown[l].contains(&id) {
        unknown[l].push(id);
      }
    }
    for lv in (a + 1..=l).rev() {
      let mut list = std::mem::take(&mut unknown[lv]);
      list.sort_unstable();
      list.dedup();
      let entries = self.read_entries(ctx, &list);
      for (&id, en) in list.iter().zip(&entries) {
        let k = [en[0], en[1], en[2], en[3]];
        keys[lv].insert(id, k);
        for c in k {
          if !self.g2h[lv - 1].contains_key(&c) {
            unknown[lv - 1].push(c);
          }
        }
      }
      unknown[lv] = list;
    }
    unknown[a].sort_unstable();
    unknown[a].dedup();
    for &t in &unknown[a] {
      let words = ctx.read_u32(&self.store, t as u64 * self.w as u64, self.w as u64);
      let h = e.tile(&words);
      self.g2h[a].insert(t, h);
      self.h2g[a].insert(h, t);
    }
    for lv in a + 1..=l {
      for &id in &unknown[lv] {
        let k = keys[lv][&id];
        let ch = k.map(|c| self.g2h[lv - 1][&c]);
        let h = e.intern(lv, ch);
        self.g2h[lv].insert(id, h);
        self.h2g[lv].insert(h, id);
      }
    }
    ids.iter().map(|id| self.g2h[l][id]).collect()
  }

  /// Entries of the given slots (eight words each), in one read-back per contiguous run.
  pub fn read_entries(&self, ctx: &GpuContext, slots: &[u32]) -> Vec<[u32; 8]> {
    if slots.is_empty() {
      return Vec::new();
    }
    // a gather by copies: one copy per slot into a staging buffer
    let n = slots.len() as u64;
    let staging = ctx.storage("gather", n * ENTRY * 4);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    let per = 1u64 << self.shard_log2;
    for (i, &s) in slots.iter().enumerate() {
      let (shard, local) = ((s as u64 / per) as usize, s as u64 % per);
      enc.copy_buffer_to_buffer(&self.nodes[shard], local * ENTRY * 4, &staging, i as u64 * ENTRY * 4, ENTRY * 4);
    }
    ctx.queue.submit(Some(enc.finish()));
    let v = ctx.read_u32(&staging, 0, n * ENTRY);
    v.chunks(8).map(|c| [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]).collect()
  }

  /// The empty node's GPU ID of every level up to `l` (uploaded from the host's).
  pub fn empties(&mut self, ctx: &GpuContext, e: &mut Engine, l: usize) -> Result<Vec<u32>> {
    while self.empty.len() <= l {
      let lv = self.empty.len().max(self.a);
      while self.empty.len() < self.a {
        self.empty.push(NONE);
      }
      let z = e.empty(lv);
      let g = self.upload(ctx, e, lv, &[z])?[0];
      self.empty.push(g);
    }
    Ok(self.empty.clone())
  }

  /// Forget the translation (the host reclaimed its nodes); the GPU tables stay.
  pub fn forget_host(&mut self) {
    self.h2g.clear();
    self.g2h.clear();
  }
}

/// The probe and insert rates on a table larger than the L2 cache, for calibration: R_P on a
/// 64 MiB table half full, R_I as a batch of 2^20 new keys.
pub fn calibrate_tables(ctx: &GpuContext) -> Result<(f64, f64, Vec<(u64, f64)>)> {
  let log2 = 21u32;
  let mut m = Memo::new(ctx, 3, 1, log2, 10)?;
  let n = 1u64 << 20;
  let mut s = 0x2545_F491u32;
  let mut rnd = || {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    s
  };
  let keys: Vec<[u32; 4]> = (0..n).map(|_| [rnd(), rnd(), rnd(), rnd()]).collect();
  let pops = vec![0u32; n as usize];
  // insert rate: the batch's claim and commit, timed without the upload
  let flat: Vec<u32> = keys.iter().flatten().copied().collect();
  let kb = ctx.storage_init("keys", &flat);
  let pb = ctx.storage_init("pops", &pops);
  let ib = ctx.storage("ids", n * 4);
  let uni = ctx.uniform("batch", &[n as u32, 7, (1u32 << log2) - 1, 0]);
  let mut e1: Vec<(u32, &wgpu::Buffer)> = vec![(0, &uni), (1, &kb), (2, &pb), (4, &ib)];
  e1.extend(m.node_entries());
  let mut e2: Vec<(u32, &wgpu::Buffer)> = vec![(0, &uni), (4, &ib)];
  e2.extend(m.node_entries());
  let bg = ctx.bind_at(&m.claim_nodes, &e1);
  let bg2 = ctx.bind_at(&m.commit_nodes, &e2);
  let (gx, gy) = grid64(n);
  let t_insert = ctx.timed(|| {
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
      let mut pass = enc.begin_compute_pass(&Default::default());
      pass.set_pipeline(&m.claim_nodes);
      pass.set_bind_group(0, &bg, &[]);
      pass.dispatch_workgroups(gx, gy, 1);
      pass.set_pipeline(&m.commit_nodes);
      pass.set_bind_group(0, &bg2, &[]);
      pass.dispatch_workgroups(gx, gy, 1);
    }
    ctx.queue.submit(Some(enc.finish()));
  });
  m.node_count += n;
  // probe rate: the same keys found again (a second claim pass finds every key committed)
  let t_probe = ctx.timed(|| {
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
      let mut pass = enc.begin_compute_pass(&Default::default());
      pass.set_pipeline(&m.claim_nodes);
      pass.set_bind_group(0, &bg, &[]);
      for _ in 0..4 {
        pass.dispatch_workgroups(gx, gy, 1);
      }
    }
    ctx.queue.submit(Some(enc.finish()));
  });
  let r_p = 4.0 * n as f64 / t_probe;
  let r_i = n as f64 / t_insert;
  Ok((r_p, r_i, vec![((ENTRY << log2) * 4, r_p)]))
}

/// The host's view of a table's home slot of a node key (for the inspector).
pub fn home_slot(key: [u32; 4], level: u32, log2: u32) -> u32 {
  node_home(key, level, (1u32 << log2) - 1)
}

#[allow(dead_code)]
fn tile_home(words: &[u32], log2: u32) -> u32 {
  tile_hash(words) & ((1u32 << log2) - 1)
}
