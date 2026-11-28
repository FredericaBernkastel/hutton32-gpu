//! Golly's extended RLE: `b`/`o` for two-state rules, `.` and `A`..`X`, `pA`..`yO` for up to 256
//! states.

use std::fmt::Write as _;

use super::Grid;
use crate::{Error, Result};

fn rle_state(out: &mut String, s: u8, two_state: bool) {
  if two_state {
    out.push(if s == 0 { 'b' } else { 'o' });
  } else if s == 0 {
    out.push('.');
  } else {
    let s = s as u32 - 1;
    if s >= 24 {
      out.push((b'p' + (s / 24 - 1) as u8) as char);
    }
    out.push((b'A' + (s % 24) as u8) as char);
  }
}

/// Golly extended RLE of `g` with the rule header `rule`; two-state rules use `b` and `o`.
pub fn write_rle(g: &Grid, rule: &str, two_state: bool) -> String {
  let mut body = String::new();
  let mut line_len = 0usize;
  let mut emit = |body: &mut String, count: usize, tok: &str| {
    let mut item = String::new();
    if count > 1 {
      write!(item, "{count}").unwrap();
    }
    item.push_str(tok);
    if line_len + item.len() > 70 {
      body.push('\n');
      line_len = 0;
    }
    line_len += item.len();
    body.push_str(&item);
  };
  let mut pending = 0usize;
  for y in 0..g.h {
    let row = &g.cells[y * g.w..(y + 1) * g.w];
    let end = row.iter().rposition(|&c| c != 0).map(|p| p + 1).unwrap_or(0);
    if end > 0 {
      if pending > 0 {
        emit(&mut body, pending, "$");
        pending = 0;
      }
      let mut x = 0;
      while x < end {
        let s = row[x];
        let mut run = 1;
        while x + run < end && row[x + run] == s {
          run += 1;
        }
        let mut tok = String::new();
        rle_state(&mut tok, s, two_state);
        emit(&mut body, run, &tok);
        x += run;
      }
    }
    pending += 1;
  }
  body.push('!');
  format!("x = {}, y = {}, rule = {}\n{}\n", g.w, g.h, rule, body)
}

/// Read Golly extended RLE: the grid and the rule named in the header (empty if none).
pub fn read_rle(text: &str) -> Result<(Grid, String)> {
  let mut lines = text.lines();
  let (mut w, mut h, mut rule) = (0usize, 0usize, String::new());
  let mut header = false;
  for line in &mut lines {
    let t = line.trim();
    if t.is_empty() || t.starts_with('#') {
      continue;
    }
    if t.starts_with('x') {
      for part in t.split(',') {
        let mut kv = part.splitn(2, '=');
        let k = kv.next().unwrap().trim();
        let v = kv.next().unwrap_or("").trim();
        match k {
          "x" => w = v.parse().map_err(|_| Error::Pattern(format!("bad RLE header: {t}")))?,
          "y" => h = v.parse().map_err(|_| Error::Pattern(format!("bad RLE header: {t}")))?,
          "rule" => rule = v.to_string(),
          _ => {}
        }
      }
      header = true;
      break;
    }
    return Err(Error::Pattern(format!("RLE without an x = .. header: {t}")));
  }
  if !header {
    return Err(Error::Pattern("RLE without a header".into()));
  }
  let mut data = String::new();
  for line in lines {
    if !line.trim_start().starts_with('#') {
      data.push_str(line.trim());
    }
  }
  // cells as (x, y, s); the header's size is a hint, the body decides
  let mut cells: Vec<(usize, usize, u8)> = Vec::new();
  let (mut x, mut y, mut count, mut prefix) = (0usize, 0usize, 0usize, 0u32);
  let (mut maxx, mut maxy) = (w, h);
  for ch in data.bytes() {
    match ch {
      b'0'..=b'9' => count = count * 10 + (ch - b'0') as usize,
      b'$' => {
        y += count.max(1);
        x = 0;
        count = 0;
      }
      b'!' => break,
      b'p'..=b'y' => prefix = (ch - b'p') as u32 + 1,
      b'.' | b'b' | b'o' | b'A'..=b'X' => {
        let s = match ch {
          b'.' | b'b' => 0,
          b'o' => 1,
          _ => prefix * 24 + (ch - b'A') as u32 + 1,
        };
        if s > 255 {
          return Err(Error::Pattern(format!("RLE state {s} out of range")));
        }
        prefix = 0;
        let n = count.max(1);
        if s != 0 {
          for i in 0..n {
            cells.push((x + i, y, s as u8));
          }
          maxx = maxx.max(x + n);
          maxy = maxy.max(y + 1);
        }
        x += n;
        count = 0;
      }
      _ => {}
    }
  }
  let mut g = Grid::new(maxx, maxy);
  for (x, y, s) in cells {
    g.set(x, y, s);
  }
  Ok((g, rule))
}

#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn round_trip() {
    let mut g = Grid::new(30, 4);
    g.set(0, 0, 1);
    g.set(5, 1, 25);
    g.set(29, 3, 255);
    let text = write_rle(&g, "Test", false);
    let (h, rule) = read_rle(&text).unwrap();
    assert_eq!(rule, "Test");
    assert_eq!(h, g);
    let mut l = Grid::new(3, 1);
    l.cells = vec![1, 1, 1];
    assert_eq!(read_rle(&write_rle(&l, "B3/S23", true)).unwrap().0, l);
  }
}
