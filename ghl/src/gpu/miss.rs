//! The miss path on the GPU (Stage 3): the host leaves the per-wave path. After every sweep of a
//! burst, its missed windows are resolved by paper Algorithm 2 run as a fixed sequence of
//! dispatches inside the burst's compute pass:
//!
//! 1. *collect*: every missed window's key is appended to a batch; the batch is found-or-inserted
//!    with the claim protocol; nodes without a result are queued (deduplicated by a tag CAS on
//!    their entry);
//! 2. *solve(l)*: above the floor, the nine overlapping sub-windows of every queued node are made
//!    and queued one level down, solved, the four windows made of their results are queued and
//!    solved, and the node of the four results is the result; at the floor, the queued windows are
//!    gathered side by side into an atlas, the atlas is stepped by the rule's plain kernel, and the
//!    central tiles are interned by content (then the nodes above them, for floors above `a + 1`);
//! 3. *publish*: results are written into the memo entries and the sweep runs again over the
//!    missed positions.
//!
//! No dispatch is sized from the GPU's counts: every stage is dispatched for a fixed capacity,
//! invocations beyond the live count return at once, and a stage that overflows its capacity sets
//! a flag; the burst is then replayed from its snapshot with larger capacities. The CPU reads the
//! counters once per burst.

use std::num::NonZeroU64;

use super::{brute::DenseKernels, context::GpuContext, memo::Memo, sweep::IdGrid};
use crate::{tile::TileGeom, Error, Result};

/// Counter slots in the arena.
const C_OVERFLOW: u32 = 1;
const C_BATCH: u32 = 2;
const C_UNRESOLVED: u32 = 3;
const C_WORK: u32 = 8; // + level
const C_MISSES: u32 = 40; // + level: misses (queued) per level over the burst
const C_INSERTS: u32 = 72; // + level: new entries per level over the burst
const C_DISTINCT: u32 = 104; // + level: distinct requests (batch elements) per level
const COUNTS: u32 = 160;

/// The parameters of one dispatch (sixteen words).
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct U {
  op: u32,
  level: u32,
  cap: u32,
  tag: u32,
  mask: u32,
  tmask: u32,
  n_cnt: u32,
  out_cnt: u32,
  work_in: u32,
  work_out: u32,
  s9: u32,
  s4: u32,
  batch: u32,
  ids: u32,
  mul: u32,
  extra: u32,
}

impl U {
  fn words(&self) -> [u32; 16] {
    [self.op, self.level, self.cap, self.tag, self.mask, self.tmask, self.n_cnt, self.out_cnt, self.work_in, self.work_out, self.s9, self.s4, self.batch, self.ids, self.mul, self.extra]
  }
}

/// The kernels of the miss path, one module; every entry point shares one explicit layout.
fn wgsl(geom: &TileGeom, atlas_ww: u32, cols: u32) -> String {
  let a = geom.a;
  let s = geom.side as u32;
  let w = geom.words() as u32;
  let wpp = geom.wpp as u32;
  let b = geom.planes;
  format!(
    r#"struct U {{ op: u32, level: u32, cap: u32, tag: u32, mask: u32, tmask: u32, n_cnt: u32, out_cnt: u32,
  work_in: u32, work_out: u32, s9: u32, s4: u32, batch: u32, ids: u32, mul: u32, extra: u32 }};
struct G {{ sw: u32, sh: u32, dw: u32, dh: u32, off: i32, mask: u32, empty: u32, level: u32 }};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> nodes: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> ar: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> tslots: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> store: array<u32>;
@group(0) @binding(5) var<storage, read_write> atlas: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> grid: array<u32>;
@group(0) @binding(7) var<storage, read_write> gcount: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> gmiss: array<u32>;
@group(0) @binding(9) var<uniform> g: G;

const A: u32 = {a}u;
const S: u32 = {s}u;
const W: u32 = {w}u;
const WPP: u32 = {wpp}u;
const B: u32 = {b}u;
const AWW: u32 = {atlas_ww}u;
const COLS: u32 = {cols}u;
const NONE: u32 = 0xFFFFFFFFu;

fn hash4(k: vec4<u32>) -> u32 {{
  var h = k.x * 0x9E3779B1u;
  h = (h ^ (h >> 15u)) + k.y * 0x85EBCA77u;
  h = (h ^ (h >> 13u)) + k.z * 0xC2B2AE3Du;
  h = (h ^ (h >> 16u)) + k.w * 0x27D4EB2Fu;
  h = (h ^ (h >> 15u)) * 0x2C1B3C6Du;
  return h ^ (h >> 12u);
}}
fn home(k: vec4<u32>, level: u32, mask: u32) -> u32 {{ return (hash4(k) ^ (level * 0x9E3779B9u)) & mask; }}
fn idx(gid: vec3<u32>) -> u32 {{ return gid.x + gid.y * 65535u * 64u; }}
fn ld(i: u32) -> u32 {{ return atomicLoad(&ar[i]); }}
fn st(i: u32, v: u32) {{ atomicStore(&ar[i], v); }}
fn child(id: u32, q: u32) -> u32 {{ return atomicLoad(&nodes[id * 8u + q]); }}
fn res(id: u32) -> u32 {{ return atomicLoad(&nodes[id * 8u + 4u]); }}
// population of a node of `level` (a tile at level A)
fn pop(id: u32, level: u32) -> u32 {{
  if (level == A) {{ return atomicLoad(&tslots[id * 4u + 2u]); }}
  return atomicLoad(&nodes[id * 8u + 7u]);
}}
fn sat_add(x: u32, y: u32) -> u32 {{ let s = x + y; if (s < x) {{ return 0xFFFFFFFFu; }} return s; }}
fn pop4(k: vec4<u32>, level: u32) -> u32 {{
  return sat_add(sat_add(pop(k.x, level), pop(k.y, level)), sat_add(pop(k.z, level), pop(k.w, level)));
}}
fn overflow(level: u32) {{ atomicStore(&ar[{C_OVERFLOW}u], 1u + level); }}

// queue `id` (level u.level) for a result unless it has one or is queued under this tag
fn enqueue(id: u32) {{
  if (res(id) != NONE) {{ return; }}
  let b = id * 8u + 6u;
  let old = atomicLoad(&nodes[b]);
  let tagged = (old & 0xFFu) | (u.tag << 8u);
  if (old == tagged) {{ return; }}
  let r = atomicCompareExchangeWeak(&nodes[b], old, tagged);
  if (!r.exchanged) {{ return; }}
  let n = atomicAdd(&ar[u.out_cnt], 1u);
  if (n >= u.cap) {{ overflow(u.level); return; }}
  st(u.work_out + n, id);
  atomicAdd(&ar[{C_MISSES}u + u.level], 1u);
}}

@compute @workgroup_size(64) fn reset(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let i = idx(gid);
  if (i >= u.cap) {{ return; }}
  if (u.op == 0u) {{ st(u.extra + i, 0u); }} else {{ atomicStore(&gcount[u.extra + i], 0u); }}
}}

fn at(x: i32, y: i32) -> u32 {{
  if (x < 0 || y < 0 || x >= i32(g.sw) || y >= i32(g.sh)) {{ return g.empty; }}
  return grid[u32(y) * g.sw + u32(x)];
}}
// the keys of the windows the sweep missed
@compute @workgroup_size(64) fn collect(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let i = idx(gid);
  let total = atomicLoad(&gcount[0]);
  if (i == 0u && total > arrayLength(&gmiss)) {{ overflow(u.level); }}
  let n = min(total, arrayLength(&gmiss));
  if (i >= n) {{ return; }}
  if (i >= u.cap) {{ overflow(u.level); return; }}
  let p = gmiss[i];
  let x = i32(p % g.dw) + g.off;
  let y = i32(p / g.dw) + g.off;
  let k = vec4<u32>(at(x, y), at(x + 1, y), at(x, y + 1), at(x + 1, y + 1));
  let j = atomicAdd(&ar[{C_BATCH}u], 1u);
  let o = u.batch + j * 5u;
  st(o, k.x); st(o + 1u, k.y); st(o + 2u, k.z); st(o + 3u, k.w);
  st(o + 4u, pop4(k, u.level - 1u));
}}

fn bkey(j: u32) -> vec4<u32> {{ let o = u.batch + j * 5u; return vec4<u32>(ld(o), ld(o + 1u), ld(o + 2u), ld(o + 3u)); }}
fn batch_n() -> u32 {{ return ld(u.n_cnt) * u.mul; }}
// find-or-insert of the batch at u.level: the claim protocol
@compute @workgroup_size(64) fn claim(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let j = idx(gid);
  if (j >= batch_n() || j >= u.cap) {{ return; }}
  let k = bkey(j);
  var slot = home(k, u.level, u.mask);
  atomicAdd(&ar[{C_DISTINCT}u + u.level], 1u);
  for (var i = 0u; i <= u.mask; i++) {{
    let b = slot * 8u;
    var c = atomicLoad(&nodes[b + 5u]);
    if (c == 0u) {{
      let r = atomicCompareExchangeWeak(&nodes[b + 5u], 0u, j + 1u);
      if (r.exchanged) {{
        atomicStore(&nodes[b], k.x); atomicStore(&nodes[b + 1u], k.y); atomicStore(&nodes[b + 2u], k.z); atomicStore(&nodes[b + 3u], k.w);
        atomicStore(&nodes[b + 4u], NONE);
        atomicStore(&nodes[b + 6u], u.level);
        atomicStore(&nodes[b + 7u], ld(u.batch + j * 5u + 4u));
        st(u.ids + j, slot | 0x80000000u);
        atomicAdd(&ar[{C_INSERTS}u + u.level], 1u);
        return;
      }}
      c = r.old_value;
    }}
    if ((c & 0x80000000u) != 0u) {{
      if (atomicLoad(&nodes[b]) == k.x && atomicLoad(&nodes[b + 1u]) == k.y && atomicLoad(&nodes[b + 2u]) == k.z
          && atomicLoad(&nodes[b + 3u]) == k.w && (atomicLoad(&nodes[b + 6u]) & 0xFFu) == u.level) {{
        st(u.ids + j, slot);
        return;
      }}
    }} else if (c != 0u) {{
      if (all(bkey(c - 1u) == k)) {{ st(u.ids + j, slot); return; }}
    }}
    slot = (slot + 1u) & u.mask;
  }}
  st(u.ids + j, 0xFFFFFFFEu);
  overflow(31u);
}}
@compute @workgroup_size(64) fn commit(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let j = idx(gid);
  if (j >= batch_n() || j >= u.cap) {{ return; }}
  let id = ld(u.ids + j);
  if (id != 0xFFFFFFFEu && (id & 0x80000000u) != 0u) {{
    let slot = id & 0x7FFFFFFFu;
    atomicStore(&nodes[slot * 8u + 5u], 0x80000000u);
    st(u.ids + j, slot);
  }}
}}
// queue the batch's nodes that have no result
@compute @workgroup_size(64) fn enqueue_batch(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let j = idx(gid);
  if (j >= batch_n() || j >= u.cap) {{ return; }}
  let id = ld(u.ids + j);
  if (id < 0xFFFFFFFEu) {{ enqueue(id); }}
}}

// ---- above the floor ----
fn witem(i: u32) -> u32 {{ return ld(u.work_in + i); }}
fn wn() -> u32 {{ return min(ld(u.n_cnt), u.cap); }}
// level u.level + 1 nodes: the five new overlapping nodes of level u.level (keys of level u.level - 1)
@compute @workgroup_size(64) fn nine_keys(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let i = idx(gid);
  if (i >= wn()) {{ return; }}
  let m = witem(i);
  let nw = child(m, 0u); let ne = child(m, 1u); let sw = child(m, 2u); let se = child(m, 3u);
  let a = vec4<u32>(child(nw, 0u), child(nw, 1u), child(nw, 2u), child(nw, 3u));
  let b = vec4<u32>(child(ne, 0u), child(ne, 1u), child(ne, 2u), child(ne, 3u));
  let c = vec4<u32>(child(sw, 0u), child(sw, 1u), child(sw, 2u), child(sw, 3u));
  let d = vec4<u32>(child(se, 0u), child(se, 1u), child(se, 2u), child(se, 3u));
  let keys = array<vec4<u32>, 5>(
    vec4<u32>(a.y, b.x, a.w, b.z), vec4<u32>(a.z, a.w, c.x, c.y), vec4<u32>(a.w, b.z, c.y, d.x),
    vec4<u32>(b.z, b.w, d.x, d.y), vec4<u32>(c.y, d.x, c.w, d.z));
  for (var q = 0u; q < 5u; q++) {{
    let o = u.batch + (i * 5u + q) * 5u;
    let k = keys[q];
    st(o, k.x); st(o + 1u, k.y); st(o + 2u, k.z); st(o + 3u, k.w);
    st(o + 4u, pop4(k, u.level - 2u));
  }}
}}
// the nine IDs of every item; the nine queued one level down
@compute @workgroup_size(64) fn nine_store(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let i = idx(gid);
  if (i >= wn()) {{ return; }}
  let m = witem(i);
  let id = u.ids + i * 5u;
  let nine = array<u32, 9>(child(m, 0u), ld(id), child(m, 1u), ld(id + 1u), ld(id + 2u), ld(id + 3u), child(m, 2u), ld(id + 4u), child(m, 3u));
  for (var q = 0u; q < 9u; q++) {{
    st(u.s9 + i * 9u + q, nine[q]);
    enqueue(nine[q]);
  }}
}}
@compute @workgroup_size(64) fn four_keys(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let i = idx(gid);
  if (i >= wn()) {{ return; }}
  var r = array<u32, 9>();
  for (var q = 0u; q < 9u; q++) {{ r[q] = res(ld(u.s9 + i * 9u + q)); }}
  let keys = array<vec4<u32>, 4>(vec4<u32>(r[0], r[1], r[3], r[4]), vec4<u32>(r[1], r[2], r[4], r[5]), vec4<u32>(r[3], r[4], r[6], r[7]), vec4<u32>(r[4], r[5], r[7], r[8]));
  for (var q = 0u; q < 4u; q++) {{
    let o = u.batch + (i * 4u + q) * 5u;
    let k = keys[q];
    st(o, k.x); st(o + 1u, k.y); st(o + 2u, k.z); st(o + 3u, k.w);
    st(o + 4u, pop4(k, u.level - 2u));
  }}
}}
@compute @workgroup_size(64) fn four_store(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let i = idx(gid);
  if (i >= wn()) {{ return; }}
  for (var q = 0u; q < 4u; q++) {{
    let id = ld(u.ids + i * 4u + q);
    st(u.s4 + i * 4u + q, id);
    enqueue(id);
  }}
}}
@compute @workgroup_size(64) fn finish_keys(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let i = idx(gid);
  if (i >= wn()) {{ return; }}
  let k = vec4<u32>(res(ld(u.s4 + i * 4u)), res(ld(u.s4 + i * 4u + 1u)), res(ld(u.s4 + i * 4u + 2u)), res(ld(u.s4 + i * 4u + 3u)));
  let o = u.batch + i * 5u;
  st(o, k.x); st(o + 1u, k.y); st(o + 2u, k.z); st(o + 3u, k.w);
  // this stage runs at the claim's level (l - 1): the key's children are one level below it
  st(o + 4u, pop4(k, u.level - 1u));
}}
// publish: the result of every item
@compute @workgroup_size(64) fn finish_store(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let i = idx(gid);
  if (i >= wn()) {{ return; }}
  atomicStore(&nodes[witem(i) * 8u + 4u], ld(u.ids + i));
}}

// ---- the floor: an atlas of windows of side 2^level ----
// zero the atlas rows of the live windows (atlas words of the ping buffer, ring included)
@compute @workgroup_size(64) fn atlas_clear(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let i = idx(gid);
  let side = 1u << u.level;
  let rows = ((wn() + COLS - 1u) / COLS) * side + 2u;
  if (i >= rows * AWW * B) {{ return; }}
  atomicStore(&atlas[i], 0u);
}}
// one invocation per base tile of a live window: walk down from the window to the tile, or its bits into the atlas
@compute @workgroup_size(64) fn atlas_gather(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let t = idx(gid);
  let per = 1u << (2u * (u.level - A));
  let i = t / per;
  if (i >= wn()) {{ return; }}
  let q = t % per;
  // tile coordinates within the window, from the quadrant path (top-level quadrant in the high bits)
  var id = witem(i);
  var tx = 0u;
  var ty = 0u;
  for (var l = u.level; l > A; l--) {{
    let sh = 2u * (l - A - 1u);
    let quad = (q >> sh) & 3u;
    id = child(id, quad);
    tx = tx * 2u + (quad & 1u);
    ty = ty * 2u + (quad >> 1u);
  }}
  let side = 1u << u.level;
  let x0 = (i % COLS) * side + tx * S;
  let y0 = (i / COLS) * side + ty * S;
  for (var k = 0u; k < B; k++) {{
    for (var y = 0u; y < S; y++) {{
      var row = 0u;
      var row_hi = 0u;
      if (S == 64u) {{ row = store[id * W + k * WPP + y * 2u]; row_hi = store[id * W + k * WPP + y * 2u + 1u]; }}
      else if (S == 32u) {{ row = store[id * W + k * WPP + y]; }}
      else {{
        let bit = y * S;
        row = (store[id * W + k * WPP + bit / 32u] >> (bit % 32u)) & ((1u << S) - 1u);
      }}
      let x = x0;
      let grp = x / 32u + 1u;
      let gy = y0 + y + 1u;
      if (row != 0u) {{ atomicOr(&atlas[(gy * AWW + grp) * B + k], row << (x % 32u)); }}
      if (S == 64u && row_hi != 0u) {{ atomicOr(&atlas[(gy * AWW + grp + 1u) * B + k], row_hi); }}
    }}
  }}
}}
// the central tiles of every live window into the tile batch (W words each)
@compute @workgroup_size(64) fn atlas_extract(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let t = idx(gid);
  let per = 1u << (2u * (u.level - 1u - A));
  let i = t / per;
  if (i >= wn()) {{ return; }}
  let q = t % per;
  let cs = 1u << (u.level - 1u - A);
  let tx = q % cs;
  let ty = q / cs;
  let side = 1u << u.level;
  let x = (i % COLS) * side + side / 4u + tx * S;
  let y0 = (i / COLS) * side + side / 4u + ty * S;
  let o = u.batch + t * W;
  for (var w = 0u; w < W; w++) {{ st(o + w, 0u); }}
  for (var k = 0u; k < B; k++) {{
    for (var y = 0u; y < S; y++) {{
      let gy = y0 + y + 1u;
      let grp = x / 32u + 1u;
      let v = atomicLoad(&atlas[(gy * AWW + grp) * B + k]);
      if (S == 64u) {{
        st(o + k * WPP + y * 2u, v);
        st(o + k * WPP + y * 2u + 1u, atomicLoad(&atlas[(gy * AWW + grp + 1u) * B + k]));
      }} else if (S == 32u) {{
        st(o + k * WPP + y, v);
      }} else {{
        let row = (v >> (x % 32u)) & ((1u << S) - 1u);
        let bit = y * S;
        let wi = o + k * WPP + bit / 32u;
        st(wi, ld(wi) | (row << (bit % 32u)));
      }}
    }}
  }}
}}
fn tw_in(i: u32) -> u32 {{ return ld(i); }}
fn tile_hash(base: u32) -> u32 {{
  var h = 0x9E3779B9u;
  for (var i = 0u; i < W; i++) {{
    h ^= tw_in(base + i);
    h = h * 0x85EBCA6Bu;
    h ^= h >> 13u;
  }}
  h ^= h >> 16u;
  h = h * 0xC2B2AE35u;
  return h ^ (h >> 16u);
}}
fn tile_pop(o: u32) -> u32 {{
  var p = 0u;
  for (var q = 0u; q < WPP; q++) {{
    var v = 0u;
    for (var k = 0u; k < B; k++) {{ v |= ld(o + k * WPP + q); }}
    p += countOneBits(v);
  }}
  return p;
}}
fn tile_equal_batch(x: u32, y: u32) -> bool {{
  for (var i = 0u; i < W; i++) {{ if (ld(u.batch + x * W + i) != ld(u.batch + y * W + i)) {{ return false; }} }}
  return true;
}}
fn tile_equal_store(slot: u32, j: u32) -> bool {{
  for (var i = 0u; i < W; i++) {{ if (store[slot * W + i] != ld(u.batch + j * W + i)) {{ return false; }} }}
  return true;
}}
fn tbatch_n() -> u32 {{ return min(ld(u.n_cnt), u.cap) * u.mul; }}
@compute @workgroup_size(64) fn tile_claim(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let j = idx(gid);
  if (j >= tbatch_n()) {{ return; }}
  let h = tile_hash(u.batch + j * W);
  var slot = h & u.tmask;
  for (var i = 0u; i <= u.tmask; i++) {{
    let b = slot * 4u;
    var c = atomicLoad(&tslots[b]);
    if (c == 0u) {{
      let r = atomicCompareExchangeWeak(&tslots[b], 0u, j + 1u);
      if (r.exchanged) {{
        atomicStore(&tslots[b + 1u], h);
        for (var q = 0u; q < W; q++) {{ store[slot * W + q] = ld(u.batch + j * W + q); }}
        atomicStore(&tslots[b + 2u], tile_pop(u.batch + j * W));
        st(u.ids + j, slot | 0x80000000u);
        atomicAdd(&ar[{C_INSERTS}u + A], 1u);
        return;
      }}
      c = r.old_value;
    }}
    if ((c & 0x80000000u) != 0u) {{
      if (atomicLoad(&tslots[b + 1u]) == h && tile_equal_store(slot, j)) {{ st(u.ids + j, slot); return; }}
    }} else if (c != 0u) {{
      if (tile_equal_batch(c - 1u, j)) {{ st(u.ids + j, slot); return; }}
    }}
    slot = (slot + 1u) & u.tmask;
  }}
  st(u.ids + j, 0xFFFFFFFEu);
  overflow(30u);
}}
@compute @workgroup_size(64) fn tile_commit(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let j = idx(gid);
  if (j >= tbatch_n()) {{ return; }}
  let id = ld(u.ids + j);
  if (id != 0xFFFFFFFEu && (id & 0x80000000u) != 0u) {{
    let slot = id & 0x7FFFFFFFu;
    atomicStore(&tslots[slot * 4u], 0x80000000u);
    st(u.ids + j, slot);
  }}
}}
// one level up within the centres: keys of 2x2 IDs of the level below (u.extra: offset of the IDs
// below, u.mul: nodes per centre at this level, side cs = sqrt(4 * mul))
@compute @workgroup_size(64) fn up_keys(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let t = idx(gid);
  let per = u.mul;
  let i = t / per;
  if (i >= wn()) {{ return; }}
  let q = t % per;
  var cs = 1u;
  while (cs * cs < per) {{ cs *= 2u; }}
  let bx = q % cs;
  let by = q / cs;
  let below = cs * 2u;
  let base = u.extra + i * below * below;
  let k = vec4<u32>(ld(base + (2u * by) * below + 2u * bx), ld(base + (2u * by) * below + 2u * bx + 1u),
                    ld(base + (2u * by + 1u) * below + 2u * bx), ld(base + (2u * by + 1u) * below + 2u * bx + 1u));
  let o = u.batch + t * 5u;
  st(o, k.x); st(o + 1u, k.y); st(o + 2u, k.z); st(o + 3u, k.w);
  st(o + 4u, pop4(k, u.level - 1u));
}}
// copy batch IDs to a scratch area (u.extra), count = live items * u.mul
@compute @workgroup_size(64) fn copy_ids(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let t = idx(gid);
  if (t >= wn() * u.mul) {{ return; }}
  st(u.extra + t, ld(u.ids + t));
}}
// add the sweep's miss counter into the unresolved count (after the re-probe)
@compute @workgroup_size(1) fn tally() {{
  atomicAdd(&ar[{C_UNRESOLVED}u], atomicLoad(&gcount[0]));
}}
"#
  )
}

/// Entry points of the module.
const ENTRIES: [&str; 17] = [
  "reset", "collect", "claim", "commit", "enqueue_batch", "nine_keys", "nine_store", "four_keys", "four_store", "finish_keys", "finish_store", "atlas_clear", "atlas_gather", "atlas_extract", "tile_claim", "tile_commit", "up_keys",
];

/// The miss path's buffers, kernels and capacities.
pub struct MissPath {
  pub geom: TileGeom,
  /// the floor: the level of the brute-forced windows
  pub floor: usize,
  /// the top level: the sweep's windows (`lt + 1`)
  pub top: usize,
  pub cap: Vec<u32>,
  arena: wgpu::Buffer,
  arena_words: u64,
  layout: wgpu::BindGroupLayout,
  pipes: Vec<wgpu::ComputePipeline>,
  copy_ids: wgpu::ComputePipeline,
  tally: wgpu::ComputePipeline,
  atlas: [wgpu::Buffer; 2],
  atlas_ww: u32,
  atlas_h: u32,
  /// arena offsets
  work_off: Vec<u32>,
  s9_off: Vec<u32>,
  s4_off: Vec<u32>,
  batch_off: u32,
  ids_off: u32,
  up_off: [u32; 2],
  batch_cap: u32,
  tag: u32,
  pub overflows: u64,
  /// the shared-tile kernel stepping the atlas in one dispatch, with its tile shape
  atlas_tile: Option<(wgpu::ComputePipeline, (u32, u32))>,
}

/// What one burst on the GPU did.
#[derive(Clone, Debug, Default)]
pub struct MissBurst {
  pub overflow: Option<u32>,
  pub unresolved: u64,
  /// per level: `(misses, inserts, distinct)`
  pub levels: Vec<(u32, u64, u64, u64)>,
  pub dispatches: u32,
}

fn pipe(ctx: &GpuContext, layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, entry: &str) -> wgpu::ComputePipeline {
  ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: Some(layout), module, entry_point: Some(entry), compilation_options: Default::default(), cache: None })
}

impl MissPath {
  /// The miss path for a sweep at tile level `lt`, floor `floor` (window level, at least `a + 1`),
  /// with `cap` items per wave and level.
  pub fn new(ctx: &GpuContext, memo: &Memo, kernels: &DenseKernels, geom: TileGeom, lt: usize, floor: usize, cap: u32) -> Result<MissPath> {
    let (shards, shard_log2) = (memo.nodes.len() as u32, memo.shard_log2);
    let a = geom.a as usize;
    let top = lt + 1;
    let floor = floor.max(a + 1).min(top);
    let caps = vec![cap; top + 1];
    let side = 1u32 << floor;
    let cols = (1024 / side).max(1);
    let atlas_ww = (cols * side).div_ceil(32) + 2;
    let atlas_h = cap.div_ceil(cols) * side + 2;
    let mut work_off = vec![0u32; top + 1];
    let mut s9_off = vec![0u32; top + 1];
    let mut s4_off = vec![0u32; top + 1];
    let mut off = COUNTS;
    for l in floor..=top {
      work_off[l] = off;
      off += caps[l];
      s9_off[l] = off;
      off += 9 * caps[l];
      s4_off[l] = off;
      off += 4 * caps[l];
    }
    let tiles_per_centre = 1u32 << (2 * (floor - 1 - a));
    let batch_cap = (5 * cap).max(cap * tiles_per_centre * (geom.words() as u32).max(5) / 5 + 1);
    let batch_off = off;
    off += batch_cap * 5.max(geom.words() as u32);
    let ids_off = off;
    off += batch_cap;
    let up = cap * tiles_per_centre;
    let up_off = [off, off + up];
    off += 2 * up;
    let arena_words = off as u64;
    if arena_words * 4 > ctx.max_binding() {
      return Err(Error::Gpu("the miss path's arena exceeds one binding".into()));
    }
    let arena = ctx.storage("miss arena", arena_words * 4);
    let b = geom.planes as u64;
    let atlas_bytes = atlas_ww as u64 * atlas_h as u64 * b * 4;
    let atlas = [ctx.storage("atlas A", atlas_bytes), ctx.storage("atlas B", atlas_bytes)];
    let st = |ty: wgpu::BufferBindingType| wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None };
    let storage = st(wgpu::BufferBindingType::Storage { read_only: false });
    let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..10).filter(|&i| i != 1).chain((0..shards).map(|j| super::memo::NODE_BINDING + j)).map(|i| wgpu::BindGroupLayoutEntry { binding: i, visibility: wgpu::ShaderStages::COMPUTE, ty: storage, count: None }).collect();
    for e in entries.iter_mut() {
      if e.binding == 0 || e.binding == 9 {
        e.ty = st(wgpu::BufferBindingType::Uniform);
      }
    }
    let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("miss path"), entries: &entries });
    let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("miss path"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
    let src = super::memo::shard_nodes(&wgsl(&geom, atlas_ww, cols), shards, shard_log2);
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let desc = wgpu::ShaderModuleDescriptor { label: Some("miss path"), source: wgpu::ShaderSource::Wgsl(src.into()) };
    // SAFETY: the library's own kernels; every index is bounded by the capacities above
    let module = unsafe { ctx.device.create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked()) };
    let pipes: Vec<wgpu::ComputePipeline> = ENTRIES.iter().map(|e| pipe(ctx, &pl, &module, e)).collect();
    let copy_ids = pipe(ctx, &pl, &module, "copy_ids");
    let tally = pipe(ctx, &pl, &module, "tally");
    if let Some(e) = pollster::block_on(scope.pop()) {
      return Err(Error::Gpu(format!("miss-path kernels: {e}")));
    }
    let _ = NonZeroU64::new(1);
    // the atlas is a small grid: all of its generations in one dispatch of a shared-tile kernel
    let gens = 1u32 << (floor as u32 - 2);
    let atlas_tile = if gens > 1 {
      let (tw, th, _) = super::brute::tile_shape(geom.planes, &ctx.limits);
      let k = gens.min(32);
      let fits = (tw + 2) * (th + 2 * k) <= ctx.limits.max_compute_invocations_per_workgroup.min(256) && 2 * (tw + 2) * (th + 2 * k) * geom.planes * 4 <= ctx.limits.max_compute_workgroup_storage_size;
      if fits {
        let src = super::kernels::shared_tile(&kernels.src, tw, th, k);
        ctx.pipeline("atlas shared tile", &src, "main", true).ok().map(|p| (p, (tw, th)))
      } else {
        None
      }
    } else {
      None
    };
    Ok(MissPath {
      geom,
      floor,
      top,
      cap: caps,
      arena,
      arena_words,
      layout,
      pipes,
      copy_ids,
      tally,
      atlas,
      atlas_ww,
      atlas_h,
      work_off,
      s9_off,
      s4_off,
      batch_off,
      ids_off,
      up_off,
      batch_cap,
      tag: 1,
      overflows: 0,
      atlas_tile,
    })
  }

  pub fn bytes(&self) -> u64 {
    self.arena_words * 4 + 2 * self.atlas_ww as u64 * self.atlas_h as u64 * self.geom.planes as u64 * 4
  }

  fn p(&self, name: &str) -> &wgpu::ComputePipeline {
    &self.pipes[ENTRIES.iter().position(|e| *e == name).unwrap()]
  }

  /// Run a burst of `n` sweeps (even) with every miss resolved on the GPU, in one compute pass;
  /// one read-back. On overflow the caller restores the snapshot and calls again: the capacities
  /// have grown.
  #[allow(clippy::too_many_arguments)]
  pub fn burst(&mut self, ctx: &GpuContext, memo: &Memo, grid: &mut IdGrid, kernels: &DenseKernels, n: u32, step_log2: u32) -> Result<MissBurst> {
    let node_mask = (1u32 << memo.node_log2) - 1;
    let tile_mask = (1u32 << memo.tile_log2) - 1;
    let a = self.geom.a as usize;
    // the uniforms of the whole schedule, written once; bind groups per dispatch
    let mut sched: Vec<(String, U, u64, u8)> = Vec::new(); // (kernel, params, invocations, grid binding: 0 none, 1 A->B, 2 B->A)
    let reset = |sched: &mut Vec<(String, U, u64, u8)>, first: u32, count: u32, which: u32| {
      sched.push(("reset".into(), U { op: which, cap: count, extra: first, ..Default::default() }, count as u64, 0));
    };
    reset(&mut sched, 0, COUNTS, 0);
    let mut tag = self.tag;
    let atlas_gens = 1u32 << ((self.floor as u32 - 2).min(step_log2));
    for sw in 0..n {
      let phase = if sw % 2 == 0 { 1u8 } else { 2u8 };
      // the sweep itself, then the collection of its misses
      sched.push(("@sweep".into(), U::default(), 0, phase));
      let top = self.top as u32;
      reset(&mut sched, C_BATCH, 1, 0);
      reset(&mut sched, C_WORK + top, 1, 0);
      sched.push(("collect".into(), U { level: top, cap: self.cap[self.top], batch: self.batch_off, ..Default::default() }, grid.miss_capacity.min(self.cap[self.top] as u64), phase));
      let claim_u = U { level: top, cap: self.batch_cap, mask: node_mask, n_cnt: C_BATCH, mul: 1, batch: self.batch_off, ids: self.ids_off, ..Default::default() };
      sched.push(("claim".into(), claim_u, self.batch_cap as u64, 0));
      sched.push(("commit".into(), claim_u, self.batch_cap as u64, 0));
      tag += 1;
      sched.push(("enqueue_batch".into(), U { tag, out_cnt: C_WORK + top, work_out: self.work_off[self.top], cap: self.cap[self.top], ..claim_u }, self.batch_cap as u64, 0));
      self.solve(&mut sched, self.top, &mut tag, node_mask, tile_mask, a, atlas_gens);
      // publish: the sweep again; what still misses is counted
      sched.push(("@resweep".into(), U::default(), 0, phase));
    }
    self.tag = tag + 1;
    // encode
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    grid.encode_snapshot(&mut enc);
    let mut keep: Vec<wgpu::BindGroup> = Vec::new();
    let mut dispatches = 0u32;
    {
      let mut pass = enc.begin_compute_pass(&Default::default());
      for (name, u, inv, phase) in &sched {
        dispatches += 1;
        match name.as_str() {
          "@sweep" | "@resweep" => {
            // reset the sweep's miss counter, sweep, and after the re-probe tally what is left
            let ab = *phase == 1;
            grid.encode_reset(&mut pass);
            grid.encode_sweep(&mut pass, ab);
            if name == "@resweep" {
              grid.encode_tick(&mut pass);
              let bg = self.bind(ctx, memo, grid, U::default(), ab);
              pass.set_pipeline(&self.tally);
              pass.set_bind_group(0, &bg, &[]);
              pass.dispatch_workgroups(1, 1, 1);
              keep.push(bg);
            }
          }
          "@atlas_step" => {
            if let Some((tp, (tw, th))) = &self.atlas_tile {
              let bgs = self.atlas_binds_with(ctx, kernels, tp);
              pass.set_pipeline(tp);
              pass.set_bind_group(0, &bgs[0], &[]);
              pass.dispatch_workgroups(self.atlas_ww.div_ceil(*tw), (self.atlas_h - 2).div_ceil(*th), 1);
              keep.extend(bgs);
            } else {
              let bgs = self.atlas_binds(ctx, kernels);
              pass.set_pipeline(&kernels.plain);
              for g in 0..u.cap {
                pass.set_bind_group(0, &bgs[(g & 1) as usize], &[]);
                pass.dispatch_workgroups((self.atlas_ww - 2).div_ceil(8), (self.atlas_h - 2).div_ceil(8), 1);
              }
              keep.extend(bgs);
            }
          }
          _ => {
            let ab = *phase != 2;
            let bg = self.bind(ctx, memo, grid, *u, ab);
            let p = if name == "copy_ids" { &self.copy_ids } else { self.p(name) };
            pass.set_pipeline(p);
            pass.set_bind_group(0, &bg, &[]);
            let (gx, gy) = super::memo::grid64((*inv).max(1));
            pass.dispatch_workgroups(gx, gy, 1);
            keep.push(bg);
          }
        }
      }
    }
    ctx.queue.submit(Some(enc.finish()));
    let c = ctx.read_u32(&self.arena, 0, COUNTS as u64);
    let mut out = MissBurst { dispatches, unresolved: c[C_UNRESOLVED as usize] as u64, ..Default::default() };
    if c[C_OVERFLOW as usize] != 0 {
      let l = c[C_OVERFLOW as usize] - 1;
      out.overflow = Some(l);
      self.overflows += 1;
      // grow every level's capacity: the counts of this burst are a lower bound
      for x in self.cap.iter_mut() {
        *x = x.saturating_mul(2);
      }
      return Ok(out);
    }
    for l in a..=self.top {
      out.levels.push((l as u32, c[(C_MISSES as usize) + l] as u64, c[(C_INSERTS as usize) + l] as u64, c[(C_DISTINCT as usize) + l] as u64));
    }
    grid.sweeps += n as u64;
    Ok(out)
  }

  /// Algorithm 2 at level `l` over the queued items of `l`, appended to the schedule.
  #[allow(clippy::too_many_arguments)]
  fn solve(&self, sched: &mut Vec<(String, U, u64, u8)>, l: usize, tag: &mut u32, node_mask: u32, tile_mask: u32, a: usize, atlas_gens: u32) {
    let lv = l as u32;
    let cap = self.cap[l];
    let base = U { level: lv, cap, mask: node_mask, tmask: tile_mask, n_cnt: C_WORK + lv, work_in: self.work_off[l], batch: self.batch_off, ids: self.ids_off, ..Default::default() };
    if l == self.floor {
      // the atlas: gather, step, extract the centres, intern them
      let per = 1u64 << (2 * (l - a));
      sched.push(("atlas_clear".into(), base, self.atlas_ww as u64 * self.atlas_h as u64 * self.geom.planes as u64, 0));
      sched.push(("atlas_gather".into(), base, cap as u64 * per, 0));
      sched.push(("@atlas_step".into(), U { cap: atlas_gens, ..base }, 0, 0));
      let centre = 1u32 << (2 * (l - 1 - a));
      // the atlas buffer the step ended in: B after the one dispatch of the shared tile, else A after
      // an even number of generations
      let fin = if self.atlas_tile.is_some() { 1 } else { atlas_gens & 1 };
      sched.push(("atlas_extract".into(), U { op: fin, ..base }, cap as u64 * centre as u64, 0));
      let tb = U { mul: centre, cap, ..base };
      sched.push(("tile_claim".into(), tb, cap as u64 * centre as u64, 0));
      sched.push(("tile_commit".into(), tb, cap as u64 * centre as u64, 0));
      // up from the tiles to the centre's node (floors above a + 1)
      let mut level = a;
      let mut count = centre;
      let mut src = 0usize;
      if level + 1 < l {
        sched.push(("copy_ids".into(), U { mul: count, extra: self.up_off[src], ..base }, cap as u64 * count as u64, 0));
      }
      while level + 1 < l {
        count /= 4;
        let ub = U { level: level as u32 + 1, mul: count, extra: self.up_off[src], ..base };
        sched.push(("up_keys".into(), ub, cap as u64 * count as u64, 0));
        let cb = U { level: level as u32 + 1, mul: count, cap: cap * count, n_cnt: C_WORK + lv, ..base };
        sched.push(("claim".into(), U { cap: cap * count, ..cb }, cap as u64 * count as u64, 0));
        sched.push(("commit".into(), U { cap: cap * count, ..cb }, cap as u64 * count as u64, 0));
        level += 1;
        if level + 1 < l {
          src ^= 1;
          sched.push(("copy_ids".into(), U { mul: count, extra: self.up_off[src], ..base }, cap as u64 * count as u64, 0));
        }
      }
      sched.push(("finish_store".into(), base, cap as u64, 0));
      return;
    }
    let below = l - 1;
    let lb = below as u32;
    let queue_below = |sched: &mut Vec<(String, U, u64, u8)>, keys: &str, store: &str, per: u32, tag: &mut u32| {
      sched.push(("reset".into(), U { op: 0, cap: 1, extra: C_WORK + lb, ..Default::default() }, 1, 0));
      sched.push((keys.into(), U { s9: self.s9_off[l], s4: self.s4_off[l], ..base }, cap as u64, 0));
      let cb = U { level: lb, mul: per, cap: cap * per, ..base };
      sched.push(("claim".into(), cb, cap as u64 * per as u64, 0));
      sched.push(("commit".into(), cb, cap as u64 * per as u64, 0));
      *tag += 1;
      sched.push((store.into(), U { level: lb, tag: *tag, out_cnt: C_WORK + lb, work_out: self.work_off[below], s9: self.s9_off[l], s4: self.s4_off[l], cap: self.cap[below], ..base }, cap as u64, 0));
    };
    queue_below(sched, "nine_keys", "nine_store", 5, tag);
    self.solve(sched, below, tag, node_mask, tile_mask, a, atlas_gens);
    queue_below(sched, "four_keys", "four_store", 4, tag);
    self.solve(sched, below, tag, node_mask, tile_mask, a, atlas_gens);
    let fb = U { level: lb, mul: 1, s4: self.s4_off[l], ..base };
    sched.push(("finish_keys".into(), fb, cap as u64, 0));
    sched.push(("claim".into(), fb, cap as u64, 0));
    sched.push(("commit".into(), fb, cap as u64, 0));
    sched.push(("finish_store".into(), base, cap as u64, 0));
  }

  fn bind(&self, ctx: &GpuContext, memo: &Memo, grid: &IdGrid, u: U, ab: bool) -> wgpu::BindGroup {
    let ub = ctx.uniform("miss stage", &u.words());
    let atlas = &self.atlas[(u.op & 1) as usize];
    let (src, gu) = grid.source(ab);
    fn e(i: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
      wgpu::BindGroupEntry { binding: i, resource: b.as_entire_binding() }
    }
    let mut entries = vec![e(0, &ub), e(2, &self.arena), e(3, &memo.tile_slots), e(4, &memo.store), e(5, atlas), e(6, src), e(7, &grid.counters), e(8, &grid.misses), e(9, gu)];
    for (i, b) in memo.node_entries() {
      entries.push(e(i, b));
    }
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.layout, entries: &entries })
  }

  fn atlas_binds_with(&self, ctx: &GpuContext, k: &DenseKernels, p: &wgpu::ComputePipeline) -> Vec<wgpu::BindGroup> {
    let even = ctx.uniform("atlas even", &[self.atlas_ww, self.atlas_h, 0, 0]);
    vec![match &k.extra {
      Some(x) => ctx.bind(p, &[&even, &self.atlas[0], &self.atlas[1], x]),
      None => ctx.bind(p, &[&even, &self.atlas[0], &self.atlas[1]]),
    }]
  }

  fn atlas_binds(&self, ctx: &GpuContext, k: &DenseKernels) -> Vec<wgpu::BindGroup> {
    let even = ctx.uniform("atlas even", &[self.atlas_ww, self.atlas_h, 0, 0]);
    let odd = ctx.uniform("atlas odd", &[self.atlas_ww, self.atlas_h, 1, 0]);
    let mk = |u: &wgpu::Buffer, s: &wgpu::Buffer, d: &wgpu::Buffer| match &k.extra {
      Some(x) => ctx.bind(&k.plain, &[u, s, d, x]),
      None => ctx.bind(&k.plain, &[u, s, d]),
    };
    vec![mk(&even, &self.atlas[0], &self.atlas[1]), mk(&odd, &self.atlas[1], &self.atlas[0])]
  }
}
