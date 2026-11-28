# ruletable2netlist

Compiles a Golly rule table (`@TABLE` in a `.rule` file, or a bare `.table`) to a netlist of NOT,
AND, OR and XOR gates that computes the next state of a cell from the bits of its neighbourhood:
the form a bit-sliced kernel evaluates, 32 or 64 cells per machine word. The netlist is checked
against the table on every neighbourhood of the rule and then made smaller without changing what
it computes. No manual step, and no dependency outside this workspace (`ca-rule`, which holds the
table parser and the netlist type).

```text
ruletable2netlist Hutton32b.rule --seconds 300 --out circuits/
rule Hutton32b | 32 states (5 bits), vonNeumann, symmetries none | 315 transitions, ... | checked on 33554432 neighbourhoods (all) | ...
```

writes `circuits/Hutton32b.netlist` (text), `.wgsl` (a function over `array<u32, bits>` per cell)
and `.rs` (the same on `u64`). As a library:

```rust
let table = ca_rule::Table::load(std::path::Path::new("Hutton32b.rule"))?;
let c = ruletable2netlist::compile(&table, &ruletable2netlist::Options { seconds: 30.0, ..Default::default() })?;
assert!(c.report.exhaustive);            // equal to the table on every neighbourhood
let wgsl = c.netlist.wgsl("step_planes", table.nbhd.table_order());
```

## What it supports

Every rule table Golly's RuleLoader reads: the von Neumann, Moore, hexagonal and one-dimensional
neighbourhoods; up to 256 states; variables, bound variables (a variable that occurs more than once
in a transition) and variables as the new state; the symmetries `none`, `rotate2/3/4/6/8` with and
without `reflect`, `reflect_horizontal`, `reflect` and `permute`; the comma-less form. Rule trees
(`@TREE`) are not read.

## How

1. **Transitions.** Each line under each symmetry becomes a set of states per cell, the groups of
   cells bound to one variable, and the result (a state, or a copy of a cell). The first that
   matches a neighbourhood applies; none: no change.
2. **The rule's own symmetry.** Many rules look the same after a quarter turn if the states are
   relabelled with it (an arrow pointing east becomes one pointing north), even when the table
   spells out every direction. The relabelling is found from the function (`--symmetry` prints
   it), and every centre state is checked on all neighbourhoods: a table may keep the symmetry for
   some centre states and break it for others. The centre states that turn into one another get
   an orientation, and one table, that of orientation 0, serves them all: its conditions are read
   through the centre's orientation, its results are turned back. A cell that many of its
   transitions test for sets the turn does not change (the cell an arrow points at) is selected
   once, bit by bit, by the orientation. A test that every neighbour shares, each for the set as
   it sees it, or every neighbour but the one at a fixed place for the orientation, is one signal
   ("an excited input from any side but the one I point at"). The transitions first decide which
   result applies, one signal per result, and the bits of the state follow from those.
3. **Simplification.** A transition is removed if those after it give the same states wherever it
   is the first to match, and a cell's set is widened (to every state where possible, which needs
   no gate) if that changes no result. Every step is checked on all the neighbourhoods it touches.
   The same is done once per bit of the state, keeping only that bit. In the table of
   orientation 0 a set that cannot grow to every state is tried grown to the set with all its
   turned images: such a set is the same whichever way the centre points.
4. **Network.** One predicate per (cell, set), by Shannon expansion over the state bits with the
   codes no state has as don't-cares. The conditions of the transitions are sums of products of
   these predicates and of the centre's bits; the divisors that many products share are found and
   built once (fast extract), which is what turns "an excited arrow from the west, or from the
   south, ..." into one signal for all the transitions that test it. The results are combined as
   a priority chain of groups, per state or per bit.
5. **Check.** The network is compared with the table on every neighbourhood: all `n^k` of them when
   there are at most 2^30 (Hutton32b: 2^25, a fraction of a second), otherwise on four million
   drawn at random plus neighbourhoods that match each transition and near misses.
   `report.exhaustive` says which.
6. **Optimisation** (only where step 5 covers every neighbourhood). Resubstitution with
   don't-cares: every node is simulated on a sample; where a node's value does not reach an output
   it is free; if a constant, another node, one or two new gates over other nodes, or a selection
   between two nodes by a third agree with it where it matters, it is replaced. A replacement is
   kept only if the whole network still equals the table on the full check; a neighbourhood that
   refutes one joins the sample. When no replacement shrinks the network it is changed and the
   sweeps go on, until the time is up. The change is one of two (`--restructure` sets the share
   of the second, 60 of 100 by default). Replacements of equal size at a third of the nodes; or
   the whole network derived again: every node is a function of at most four to six nodes below
   it (a cut), cuts that cover the outputs with little logic are chosen, and each chosen node is
   built from its truth table alone, by decomposition over one variable at a time (one time in
   three the trees of AND nodes and of XOR nodes are paired again instead). That keeps
   the function on every input and gives the sweeps other signals to work with; resubstitution
   alone stops at networks that it leaves.
7. **Gates.** AND nodes become AND or OR gates and XOR nodes pick input polarities so that the
   fewest inverters are needed.

`--style fold|factor|chain|bits|flat|cover` builds in one way only; by default the three smallest
networks as built are optimised for a while and the best of them for the rest of the time.

## Limits

A rule with more than 2^30 neighbourhoods (17 states on the Moore neighbourhood, say) cannot be
checked on all of them. Its network is built as for any rule, follows the table's transitions one
to one and is tested on a sample, but the optimiser is not run on it: a replacement that is right on
every sampled neighbourhood can be wrong on others, and for one such rule an optimised netlist
passed the sample and then differed from the table on a random soup. `report.optimised` and
`report.exhaustive` say what was done. `Options { raw: true, .. }` skips the optimiser for any
rule.

The optimiser is a local search bounded by time: two runs give different netlists, a longer run
gives a smaller one, and so does a further run that starts from the result (`--from` with another
`--seed`). For Hutton32b one run of 40 minutes from the table gave 367 gates, and runs of ten
minutes from results of that kind gave the 353 gates of the circuit kept in
`research/bitslice/compiled/`; a circuit designed by hand around the meaning of the states and
then optimised with resubstitution and ABC has 367. The times are those of a loaded machine and
say nothing more than the order of magnitude.

## Checking a netlist

`--check FILE` compares a netlist written earlier (or elsewhere) with the table, without changing
it: every neighbourhood where they can be enumerated. `--lut FILE` compares with a table of next
states (one byte per neighbourhood, the centre the highest digit), and `--golly BGOLLY` runs a
random soup for 16 generations by the table, by the netlist and by Golly and compares the three.
