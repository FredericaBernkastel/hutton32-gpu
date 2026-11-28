//! WGSL kernels, generated from the rule before compilation, as the harness does: what depends on
//! the rule (its network, table or tree, the bit-planes, the generations per dispatch, the tile
//! shape) is substituted into the source, so changing the rule is a new shader module.
//!
//! Dense grids are stored as groups of `B` `u32` words per 32 cells (`B` = the rule's bit-planes),
//! word k of a group holding bit k of the states of its 32 cells (bit i = cell i). Rows of `ww`
//! groups, `h` rows, with a ring of zero groups and rows around the cells; the ring stays zero.

use std::fmt::Write as _;

use crate::{
  rule::{Pos, Rule, RuleTree},
  sched::KernelClass,
};

/// The rule as WGSL: `fn rule9(gen: u32, c, n, ne, e, se, s, sw, w, nw: array<u32, B>) -> array<u32, B>`.
#[derive(Clone, Debug)]
pub struct RuleSource {
  pub planes: u32,
  /// the positions the step reads
  pub positions: Vec<Pos>,
  pub step: String,
  /// a third storage binding: the table (bytes packed four per word) or the tree
  pub extra: Option<Vec<u32>>,
  pub class: KernelClass,
  pub name: String,
}

fn zero(b: u32) -> String {
  format!("array<u32, {b}>({})", vec!["0u"; b as usize].join(", "))
}

/// The WGSL of a rule's step on 32 cells at once.
pub fn rule_source(rule: &Rule) -> RuleSource {
  let b = rule.planes();
  let all = "gen: u32, c: array<u32, B>, n: array<u32, B>, ne: array<u32, B>, e: array<u32, B>, se: array<u32, B>, s: array<u32, B>, sw: array<u32, B>, w: array<u32, B>, nw: array<u32, B>".replace('B', &b.to_string());
  if let Some(net) = rule.network() {
    let mut positions = net.positions();
    let mut src = net.wgsl("rule_even", &positions);
    let args = |ps: &[Pos]| ps.iter().map(|p| p.name()).collect::<Vec<_>>().join(", ");
    let body = if rule.alternates() {
      let odd = rule.for_generation(1).network().expect("the odd rule of a Life-like rule has a network");
      let po = odd.positions();
      src += &odd.wgsl("rule_odd", &po);
      for p in &po {
        if !positions.contains(p) {
          positions.push(*p);
        }
      }
      format!("  if ((gen & 1u) == 0u) {{ return rule_even({}); }}\n  return rule_odd({});\n", args(&net.positions()), args(&po))
    } else {
      format!("  return rule_even({});\n", args(&positions))
    };
    src += &format!("fn rule9({all}) -> array<u32, {b}> {{\n{body}}}\n");
    let class = if rule.life_like_masks().is_some() { KernelClass::LifeLike } else { KernelClass::Circuit };
    let name = match class {
      KernelClass::LifeLike => "adder tree".to_string(),
      _ => format!("circuit ({} gates)", net.gates().len()),
    };
    return RuleSource { planes: b, positions, step: src, extra: None, class, name };
  }
  let nbhd = rule.neighbourhood();
  let fields = nbhd.fields();
  // the state of cell `i` at position `p`: bit k of it is bit i of plane k
  let state = |p: Pos| -> String { (0..b).map(|k| format!("(((({}[{k}] >> i) & 1u) << {k}u))", p.name())).collect::<Vec<_>>().join(" | ") };
  let scatter = |v: &str| -> String { (0..b).map(|k| format!("    o[{k}] |= (({v} >> {k}u) & 1u) << i;\n")).collect::<String>() };
  if let Some(table) = rule.table() {
    let mut words = vec![0u32; table.len().div_ceil(4)];
    for (i, &v) in table.iter().enumerate() {
      words[i / 4] |= (v as u32) << ((i % 4) * 8);
    }
    let mut idx = String::new();
    for (f, &p) in fields.iter().enumerate() {
      let _ = write!(idx, "    idx |= ({}) << {}u;\n", state(p), f as u32 * b);
    }
    let step = format!(
      "fn lut_get(i: u32) -> u32 {{ return (extra[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu; }}\nfn rule9({all}) -> array<u32, {b}> {{\n  var o = {z};\n  for (var i = 0u; i < 32u; i++) {{\n    var idx = 0u;\n{idx}    let v = lut_get(idx);\n{sc}  }}\n  return o;\n}}\n",
      z = zero(b),
      sc = scatter("v")
    );
    return RuleSource { planes: b, positions: fields.to_vec(), step, extra: Some(words), class: KernelClass::ByteLut, name: "byte LUT".into() };
  }
  let tree: &RuleTree = rule.tree().expect("a rule has a table or a tree");
  let order = RuleTree::order(tree.neighbours());
  let mut walk = String::new();
  for &p in order {
    let _ = write!(walk, "    node = extra[node * {st}u + ({})];\n", state(p), st = tree.states());
  }
  let step = format!(
    "fn rule9({all}) -> array<u32, {b}> {{\n  var o = {z};\n  for (var i = 0u; i < 32u; i++) {{\n    var node = {root}u;\n{walk}{sc}  }}\n  return o;\n}}\n",
    z = zero(b),
    root = tree.root(),
    sc = scatter("node")
  );
  RuleSource { planes: b, positions: nbhd.fields().to_vec(), step, extra: Some(tree.data().to_vec()), class: KernelClass::Tree, name: "rule tree".into() }
}

const HEAD: &str = "struct P { ww: u32, h: u32, gen: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
";

fn extra_binding(r: &RuleSource) -> &'static str {
  if r.extra.is_some() {
    "@group(0) @binding(3) var<storage, read> extra: array<u32>;\n"
  } else {
    ""
  }
}

/// `arr(name, base)`: `array<u32, B>(name[base + 0u], ...)`.
fn arr(b: u32, name: &str, base: &str) -> String {
  format!("array<u32, {b}>({})", (0..b).map(|k| format!("{name}[{base} + {k}u]")).collect::<Vec<_>>().join(", "))
}

/// The east and west neighbour words: cell i's east neighbour is cell i + 1.
fn shifts(b: u32) -> String {
  let e = (0..b).map(|k| format!("(c[{k}] >> 1u) | (r[{k}] << 31u)")).collect::<Vec<_>>().join(", ");
  let w = (0..b).map(|k| format!("(c[{k}] << 1u) | (l[{k}] >> 31u)")).collect::<Vec<_>>().join(", ");
  format!("fn east(c: array<u32, {b}>, r: array<u32, {b}>) -> array<u32, {b}> {{ return array<u32, {b}>({e}); }}\nfn west(c: array<u32, {b}>, l: array<u32, {b}>) -> array<u32, {b}> {{ return array<u32, {b}>({w}); }}\n")
}

/// Which neighbour words a step needs: up row, down row, left and right groups.
struct Needs {
  up: bool,
  down: bool,
  left: bool,
  right: bool,
  ul: bool,
  ur: bool,
  dl: bool,
  dr: bool,
}

fn needs(ps: &[Pos]) -> Needs {
  let has = |p: Pos| ps.contains(&p);
  Needs {
    up: has(Pos::N) || has(Pos::NE) || has(Pos::NW),
    down: has(Pos::S) || has(Pos::SE) || has(Pos::SW),
    left: has(Pos::W),
    right: has(Pos::E),
    ul: has(Pos::NW),
    ur: has(Pos::NE),
    dl: has(Pos::SW),
    dr: has(Pos::SE),
  }
}

/// The nine arguments of `rule9` from the loaded words, zero where the step reads nothing.
fn nine_args(b: u32, ps: &[Pos]) -> String {
  let z = zero(b);
  let a = |p: Pos, expr: String| if ps.contains(&p) { expr } else { z.clone() };
  [
    a(Pos::C, "c".into()),
    a(Pos::N, "u".into()),
    a(Pos::NE, "east(u, ur)".into()),
    a(Pos::E, "east(c, r)".into()),
    a(Pos::SE, "east(d, dr)".into()),
    a(Pos::S, "d".into()),
    a(Pos::SW, "west(d, dl)".into()),
    a(Pos::W, "west(c, l)".into()),
    a(Pos::NW, "west(u, ul)".into()),
  ]
  .join(", ")
}

/// The plain kernel: one invocation per group of 32 cells, 8×8 workgroups, ping-pong buffers.
pub fn plain(r: &RuleSource) -> String {
  let b = r.planes;
  let nd = needs(&r.positions);
  let z = zero(b);
  let ld = |name: &str, cond: bool, idx: &str| if cond { format!("  let {name} = ld({idx});\n") } else { format!("  let {name} = {z};\n") };
  let mut s = String::from(HEAD);
  s += extra_binding(r);
  s += &r.step;
  s += &shifts(b);
  s += &format!("fn ld(i: u32) -> array<u32, {b}> {{ let b = i * {b}u; return {}; }}\n", arr(b, "src", "b"));
  s += "@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n  let x = id.x + 1u;\n  let y = id.y + 1u;\n  if (x >= p.ww - 1u || y >= p.h - 1u) { return; }\n  let i = y * p.ww + x;\n  let c = ld(i);\n";
  s += &ld("u", nd.up, "i - p.ww");
  s += &ld("d", nd.down, "i + p.ww");
  s += &ld("l", nd.left || nd.ul || nd.dl, "i - 1u");
  s += &ld("r", nd.right || nd.ur || nd.dr, "i + 1u");
  s += &ld("ul", nd.ul, "i - p.ww - 1u");
  s += &ld("ur", nd.ur, "i - p.ww + 1u");
  s += &ld("dl", nd.dl, "i + p.ww - 1u");
  s += &ld("dr", nd.dr, "i + p.ww + 1u");
  s += &format!("  let o = rule9(p.gen, {});\n  let ob = i * {b}u;\n", nine_args(b, &r.positions));
  for k in 0..b {
    let _ = writeln!(s, "  dst[ob + {k}u] = o[{k}];");
  }
  s += "}\n";
  s
}

/// The shared-tile kernel: a workgroup loads a tile of `tw` groups by `th` rows with a halo of one
/// group and `k` rows into workgroup memory, advances it `k` generations there, and writes the tile
/// once. Bus traffic per generation falls by `k`; the halo costs circuit evaluations.
pub fn shared_tile(r: &RuleSource, tw: u32, th: u32, k: u32) -> String {
  let b = r.planes;
  let (ew, eh) = (tw + 2, th + 2 * k);
  let n = ew * eh * b;
  let stride = ew * b;
  let nd = needs(&r.positions);
  let z = zero(b);
  let mut s = String::from(HEAD);
  s += extra_binding(r);
  s += &r.step;
  s += &shifts(b);
  let _ = write!(
    s,
    "var<workgroup> sa: array<u32, {n}>;
var<workgroup> sb: array<u32, {n}>;
@compute @workgroup_size({ew}, {eh}) fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
  let lx = lid.x;
  let ly = lid.y;
  let gx = i32(wid.x * {tw}u + lx);
  let gy = i32(wid.y * {th}u + ly) + 1 - {k};
  let inside = gx >= 1 && gx < i32(p.ww) - 1 && gy >= 1 && gy < i32(p.h) - 1;
  let gi = (u32(max(gy, 0)) * p.ww + u32(max(gx, 0))) * {b}u;
  let li = (ly * {ew}u + lx) * {b}u;
  var v = {z};
  if (inside) {{ v = {src}; }}
",
    src = arr(b, "src", "gi")
  );
  for kk in 0..b {
    let _ = writeln!(s, "  sa[li + {kk}u] = v[{kk}];");
  }
  s += "  workgroupBarrier();\n";
  for g in 1..=k {
    let (from, to) = if g % 2 == 1 { ("sa", "sb") } else { ("sb", "sa") };
    let ld = |name: &str, cond: bool, base: String| if cond { format!("      let {name} = {};\n", arr(b, from, &base)) } else { format!("      let {name} = {z};\n") };
    let side = |name: &str, cond: bool, base: String, guard: &str| {
      if cond {
        format!("      var {name} = {z};\n      if ({guard}) {{ {name} = {}; }}\n", arr(b, from, &base))
      } else {
        format!("      let {name} = {z};\n")
      }
    };
    let _ = write!(s, "  {{\n    var o = {z};\n    if (inside && ly >= {g}u && ly + {g}u < {eh}u) {{\n      let c = {};\n", arr(b, from, "li"));
    s += &ld("u", nd.up, format!("li - {stride}u"));
    s += &ld("d", nd.down, format!("li + {stride}u"));
    s += &side("l", nd.left || nd.ul || nd.dl, format!("li - {b}u"), "lx > 0u");
    s += &side("r", nd.right || nd.ur || nd.dr, format!("li + {b}u"), &format!("lx + 1u < {ew}u"));
    s += &side("ul", nd.ul, format!("li - {stride}u - {b}u"), "lx > 0u");
    s += &side("ur", nd.ur, format!("li - {stride}u + {b}u"), &format!("lx + 1u < {ew}u"));
    s += &side("dl", nd.dl, format!("li + {stride}u - {b}u"), "lx > 0u");
    s += &side("dr", nd.dr, format!("li + {stride}u + {b}u"), &format!("lx + 1u < {ew}u"));
    let _ = writeln!(s, "      o = rule9(p.gen + {}u, {});\n    }}", g - 1, nine_args(b, &r.positions));
    for kk in 0..b {
      let _ = writeln!(s, "    {to}[li + {kk}u] = o[{kk}];");
    }
    s += "  }\n  workgroupBarrier();\n";
  }
  let fin = if k % 2 == 1 { "sb" } else { "sa" };
  let _ = writeln!(s, "  if (inside && lx >= 1u && lx <= {tw}u && ly >= {k}u && ly < {last}u) {{", last = th + k);
  for kk in 0..b {
    let _ = writeln!(s, "    dst[gi + {kk}u] = {fin}[li + {kk}u];");
  }
  s += "  }\n}\n";
  s
}

/// Per interior row: live cells, the first live cell's column and one past the last
/// (`0xFFFFFFFF, 0` for an empty row), three words per row.
pub fn census(b: u32) -> String {
  let or = (0..b).map(|k| format!("src[g + {k}u]")).collect::<Vec<_>>().join(" | ");
  format!(
    "struct P {{ ww: u32, h: u32, gen: u32, pad: u32 }};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
  let y = id.x + 1u;
  if (y >= p.h - 1u) {{ return; }}
  var pop = 0u;
  var first = 0xFFFFFFFFu;
  var last = 0u;
  for (var x = 1u; x < p.ww - 1u; x++) {{
    let g = (y * p.ww + x) * {b}u;
    let v = {or};
    if (v != 0u) {{
      pop += countOneBits(v);
      first = min(first, (x - 1u) * 32u + firstTrailingBit(v));
      last = max(last, (x - 1u) * 32u + firstLeadingBit(v) + 1u);
    }}
  }}
  out[(y - 1u) * 3u] = pop;
  out[(y - 1u) * 3u + 1u] = first;
  out[(y - 1u) * 3u + 2u] = last;
}}
"
  )
}

/// Copy a grid into a larger one, its interior shifted by `(ox, oy)` groups and rows.
pub fn blit(b: u32) -> String {
  format!(
    "struct Q {{ ww_s: u32, h_s: u32, ww_d: u32, h_d: u32, ox: u32, oy: u32, pad0: u32, pad1: u32 }};
@group(0) @binding(0) var<uniform> q: Q;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
  let x = id.x;
  let y = id.y;
  if (x >= q.ww_d || y >= q.h_d) {{ return; }}
  let sx = i32(x) - i32(q.ox);
  let sy = i32(y) - i32(q.oy);
  let inside = sx >= 1 && sy >= 1 && sx < i32(q.ww_s) - 1 && sy < i32(q.h_s) - 1;
  for (var k = 0u; k < {b}u; k++) {{
    var v = 0u;
    if (inside) {{ v = src[(u32(sy) * q.ww_s + u32(sx)) * {b}u + k]; }}
    dst[(y * q.ww_d + x) * {b}u + k] = v;
  }}
}}
"
  )
}

/// An empty kernel, for the latency calibration.
pub const NOP: &str = "@group(0) @binding(0) var<storage, read_write> d: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x == 0xFFFFFFFFu) { d[0] = 1u; }
}
";
