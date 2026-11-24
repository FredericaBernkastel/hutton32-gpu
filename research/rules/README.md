# Canonical rule tables

Byte-for-byte copies of the two tables that define the rules (`paper.md`, Appendix B):

| File | Rule | Upstream |
|---|---|---|
| `Hutton32.table` | Hutton32 (Tim Hutton) | http://gollygang.github.io/ruletablerepository/downloads/Hutton32.table |
| `Hutton32b.rule` | Hutton32b | https://raw.githubusercontent.com/FredericaBernkastel/Hutton32-Computer/master/doc/Hutton32b.rule |

The harness reads them as `table:rules/Hutton32.table` and `table:rules/Hutton32b.rule`. This folder also serves as the rule folder for `bgolly` (`-s rules/`) in `tools/time_golly.py` and `tools/head2head.py`; Golly's built-in Hutton32 (`-a JvN`) needs no file.
