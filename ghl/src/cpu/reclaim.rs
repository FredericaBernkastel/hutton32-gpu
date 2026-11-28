//! Reclaiming the host's memory: mark what is reachable from the nodes still in use, rebuild the
//! tile store and every level's table with only those, and hand back the renumbering. The
//! reference engine never reclaims and grew to 148 M nodes (about 7 GB) on the replicator.
//!
//! The memo is a cache: a result whose node is dropped is dropped with it, which costs a relearning
//! transient, never correctness. With `keep_results` the results of reachable nodes are marked as
//! well, as Golly's collector does, so that the work stays memoized.

use rustc_hash::FxHashMap;

use super::hl::{Engine, Level};
use crate::tile::NONE;

/// Old ID to new ID, per level (`NONE` for a dropped node). Tiles are level `a`.
pub struct Remap {
  pub a: usize,
  pub levels: Vec<Vec<u32>>,
}

impl Remap {
  /// The new ID of node `(l, id)`; panics if it was dropped.
  pub fn node(&self, l: usize, id: u32) -> u32 {
    let n = self.levels[l][id as usize];
    assert!(n != NONE, "node ({l}, {id}) was reclaimed while in use");
    n
  }
  pub fn get(&self, l: usize, id: u32) -> Option<u32> {
    self.levels.get(l).and_then(|v| v.get(id as usize)).copied().filter(|&n| n != NONE)
  }
}

/// What a reclaim did.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReclaimStats {
  pub tiles_before: u64,
  pub tiles_after: u64,
  pub nodes_before: u64,
  pub nodes_after: u64,
}

/// Keep only the nodes reachable from `roots` (pairs of level and ID), and the empty nodes.
pub fn reclaim(e: &mut Engine, roots: &[(usize, u32)], keep_results: bool) -> (Remap, ReclaimStats) {
  let a = e.a;
  let top = e.levels.len();
  let mut marked: Vec<Vec<bool>> = (0..top).map(|l| if l < a { Vec::new() } else if l == a { vec![false; e.num_tiles()] } else { vec![false; e.levels[l].nodes.len()] }).collect();
  marked[a][0] = true;
  // the empty nodes that exist are kept
  for (l, &z) in e.empty_ids().iter().enumerate() {
    if l >= a && z != NONE {
      marked[l][z as usize] = true;
    }
  }
  for &(l, id) in roots {
    if l >= a && (id as usize) < marked[l].len() {
      marked[l][id as usize] = true;
    }
  }
  // one pass from the top: a marked node marks its children (and its result) one level down
  for l in (a + 1..top).rev() {
    for id in 0..e.levels[l].nodes.len() {
      if !marked[l][id] {
        continue;
      }
      for &c in &e.levels[l].nodes[id] {
        marked[l - 1][c as usize] = true;
      }
      if keep_results {
        let r = e.levels[l].res[id];
        if r != NONE {
          marked[l - 1][r as usize] = true;
        }
      }
    }
  }
  let mut stats = ReclaimStats { tiles_before: e.num_tiles() as u64, nodes_before: e.total_nodes() as u64, ..Default::default() };
  let mut levels: Vec<Vec<u32>> = vec![Vec::new(); top];
  // tiles: keep the order, rebuild the store
  let w = e.geom.words();
  let mut tmap = vec![NONE; e.num_tiles()];
  let old_tiles: Vec<u32> = e.tile_store().to_vec();
  let old_n = e.num_tiles();
  let mut fresh = Engine::new(e.rule(), a, e.step_log2());
  fresh.set_floor(e.floor());
  for (t, m) in tmap.iter_mut().enumerate().take(old_n) {
    if marked[a][t] {
      *m = fresh.tile(&old_tiles[t * w..(t + 1) * w]);
    }
  }
  levels[a] = tmap;
  // nodes, level by level, in the order of their old IDs
  for l in a + 1..top {
    let old = std::mem::take(&mut e.levels[l]);
    let mut map = vec![NONE; old.nodes.len()];
    let mut lv = Level { map: FxHashMap::default(), stats: old.stats, ..Default::default() };
    for id in 0..old.nodes.len() {
      if !marked[l][id] {
        continue;
      }
      let c = old.nodes[id];
      let nc = [levels[l - 1][c[0] as usize], levels[l - 1][c[1] as usize], levels[l - 1][c[2] as usize], levels[l - 1][c[3] as usize]];
      let nid = lv.nodes.len() as u32;
      lv.nodes.push(nc);
      lv.res.push(NONE);
      lv.pop.push(old.pop[id]);
      lv.map.insert(nc, nid);
      map[id] = nid;
    }
    // results: kept when the result node survived
    for id in 0..old.nodes.len() {
      if map[id] != NONE && old.res[id] != NONE {
        let r = levels[l - 1][old.res[id] as usize];
        lv.res[map[id] as usize] = r;
      }
    }
    levels[l] = map;
    e.levels[l] = lv;
  }
  e.replace_tiles(fresh);
  e.remap_empty(&levels);
  stats.tiles_after = e.num_tiles() as u64;
  stats.nodes_after = e.total_nodes() as u64;
  (Remap { a, levels }, stats)
}

impl Engine {
  /// Take over the tile store of `fresh` (same geometry).
  pub(crate) fn replace_tiles(&mut self, fresh: Engine) {
    let (tiles, pop, slots) = fresh.into_tile_parts();
    self.set_tile_parts(tiles, pop, slots);
  }
}
