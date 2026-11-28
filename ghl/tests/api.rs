//! The interface of roadmap Sections 7 and 8 beyond stepping: time budgets, observers on other
//! threads, maps, traces, rule files.

use std::{path::PathBuf, time::Duration};

use ghl::{inspect::Tile, io::read_rle, Config, Pattern, Rect, Rule, Universe};

fn load(rel: &str) -> (ghl::Grid, Rule) {
  let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../research/patterns").join(rel);
  let (g, rule) = read_rle(&std::fs::read_to_string(path).unwrap()).unwrap();
  (g, Rule::named(if rule.is_empty() { "Hutton32b" } else { &rule }).unwrap())
}

/// `step_for` runs whole epochs within the budget and ends where `step` of as many generations does.
#[test]
fn step_for_equals_step() {
  let (g, rule) = load("misc/chaotic.rle");
  let mut u = Universe::new(rule.clone(), Config::cpu()).unwrap();
  u.load(&Pattern::Cells(g.clone()), (0, 0)).unwrap();
  let r = u.step_for(Duration::from_millis(30));
  assert!(r.generations > 0);
  assert_eq!(u.generation(), r.generations);
  let mut v = Universe::new(rule, Config::cpu()).unwrap();
  v.load(&Pattern::Cells(g), (0, 0)).unwrap();
  v.step(r.generations);
  assert_eq!(u.fingerprint(), v.fingerprint());
}

/// An observer on another thread reads counters and maps while the universe steps; maps asked for
/// during a step are answered at the next epoch boundary, the others at once.
#[test]
fn observer_on_another_thread() {
  let (g, rule) = load("presets/hutton32b_8bit_counter.rle");
  let mut u = Universe::new(rule, Config::cpu_step(6)).unwrap();
  u.load(&Pattern::Cells(g), (0, 0)).unwrap();
  let obs = u.observer();
  let reader = std::thread::spawn(move || {
    let mut answered = 0;
    for _ in 0..50 {
      let p = obs.ids(Rect::new(0, 0, 256, 256), 3);
      if p.wait_timeout(Duration::from_secs(10)).is_some() {
        answered += 1;
      }
      let _ = obs.sample();
      std::thread::sleep(Duration::from_millis(1));
    }
    (answered, obs.history().len())
  });
  for _ in 0..40 {
    u.step(64);
  }
  u.serve();
  let (answered, history) = reader.join().unwrap();
  assert_eq!(answered, 50, "every map request was answered");
  assert!(history >= 40, "a sample per epoch: {history}");
  // at once on a universe that is not stepping
  let obs = u.observer();
  let ids = obs.ids(Rect::new(0, 0, 64, 64), 3).poll().expect("answered at once");
  assert_eq!(ids.len(), 64);
  let id = ids.iter().copied().find(|&i| i != obs.level_info(3).empty.unwrap()).unwrap();
  assert!(matches!(obs.tile(3, id).poll().unwrap(), Some(Tile::Cells { side: 8, .. })));
  let planes = obs.read_planes(5, 5).poll().unwrap();
  assert_eq!(planes.words.len(), 5);
  // the network's signal trace of that word: its outputs are the next generation of its cells
  let rule = u.rule();
  let net = rule.network().unwrap();
  let t = net.trace(&net.trace_inputs(&planes.words));
  let next: Vec<u8> = (0..32).map(|i| (0..5).fold(0u8, |c, k| c | (((net.signal(&t, net.outputs()[k]) >> i) & 1) as u8) << k)).collect();
  u.step(1);
  let mut cells = vec![0u8; 32];
  u.read_region(Rect::new(planes.x, planes.y, 32, 1), &mut cells);
  assert_eq!(next, cells, "the traced network steps the word");
  let st = obs.memo_stats(5);
  assert!(st.entries > 0 && st.slots >= st.entries);
  let trace = u.trace_step();
  assert!(!trace.waves.is_empty());
  let sample = obs.sample();
  let totals = trace.totals();
  for lc in &sample.levels {
    if let Some(t) = totals.iter().find(|t| t.0 == lc.level) {
      assert_eq!((t.1, t.2), (lc.distinct, lc.misses), "the trace's waves sum to the sample's counters at level {}", lc.level);
    }
  }
}

/// Rule files: `@TABLE` with symmetries and variables, `@TREE`, and a table compiled into a network.
#[test]
fn rule_files() {
  let dir = std::env::temp_dir().join(format!("ghl-rules-{}", std::process::id()));
  std::fs::create_dir_all(&dir).unwrap();
  // WireWorld as a table
  let ww = "@RULE WireWorldT\n@TABLE\nn_states:4\nneighborhood:Moore\nsymmetries:permute\nvar a={0,1,2,3}\nvar b={0,1,2,3}\nvar c={0,1,2,3}\nvar d={0,1,2,3}\nvar e={0,1,2,3}\nvar f={0,1,2,3}\nvar g={0,1,2,3}\nvar h={0,2,3}\nvar i={0,2,3}\nvar j={0,2,3}\nvar k={0,2,3}\nvar l={0,2,3}\nvar m={0,2,3}\nvar n={0,2,3}\n1,a,b,c,d,e,f,g,h,2\n2,a,b,c,d,e,f,g,h,3\n3,1,h,i,j,k,l,m,n,1\n3,1,1,i,j,k,l,m,n,1\n";
  let p = dir.join("WireWorldT.rule");
  std::fs::write(&p, ww).unwrap();
  let rule = Rule::load(&p).unwrap();
  assert_eq!((rule.name(), rule.states()), ("WireWorldT", 4));
  // an electron runs along a wire
  let mut u = Universe::new(rule.clone(), Config::cpu()).unwrap();
  let mut cells: Vec<((i64, i64), u8)> = (0..20).map(|x| ((x, 0), 3u8)).collect();
  cells[1].1 = 1;
  cells[0].1 = 2;
  u.set_cells(&cells);
  u.step(5);
  let mut row = vec![0u8; 20];
  u.read_region(Rect::new(0, 0, 20, 1), &mut row);
  assert_eq!(&row[5..8], &[2, 1, 3]);
  #[cfg(feature = "compile")]
  {
    let c = Rule::load_compiled(&p, 0.0).unwrap();
    assert_eq!(c.network().unwrap().verify(&c), 0);
    let mut v = Universe::new(c, Config::cpu()).unwrap();
    v.set_cells(&cells);
    v.step(5);
    assert_eq!(v.fingerprint(), u.fingerprint());
  }
  // a tree rule: Life as @TREE, round-tripped through the library's own writer
  let life = Rule::named("B3/S23").unwrap();
  let tree = ghl::rule::RuleTree::from_fn(2, 8, &mut |v: &[u8]| {
    let n: u8 = v[..8].iter().sum();
    (n == 3 || (v[8] == 1 && n == 2)) as u8
  });
  let tp = dir.join("LifeTree.tree");
  std::fs::write(&tp, tree.to_text()).unwrap();
  let t = Rule::load(&tp).unwrap();
  assert_eq!(t.table(), life.table());
  std::fs::remove_dir_all(&dir).ok();
}
