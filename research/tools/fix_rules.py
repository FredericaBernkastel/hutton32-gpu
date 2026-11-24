"""Correct the archived WGSL rule ports in place, so that they equal the canonical rules:

    archive/src/kernel/rules/hutton32.rule.wgsl    Hutton32: Tim Hutton's table,
        http://gollygang.github.io/ruletablerepository/downloads/Hutton32.table (same as Golly's built-in Hutton32)
    archive/src/kernel/rules/hutton32b.rule.wgsl   Hutton32b:
        https://github.com/FredericaBernkastel/Hutton32-Computer/blob/master/doc/Hutton32b.rule

As archived in commit 256ef91 the two files differed from those tables in 2 430 000 and 3 082 794
of the 2^25 neighbourhoods (paper.md, section 10). The script is idempotent: a patch that is already
in the file is reported and skipped, anything else fails loudly.

Check afterwards (from research/):
    ca-research lut --rule hutton32  --against table:<Hutton32.table>     # 0 mismatches
    ca-research lut --rule hutton32b --against table:<Hutton32b.rule>     # 0 mismatches

usage: python tools/fix_rules.py        (from research/)
"""
import io
import pathlib
import re

RULES = pathlib.Path(__file__).resolve().parent.parent.parent / "archive" / "src" / "kernel" / "rules"


def patch(text, old, new, count, what, regex=False):
    found = len(re.findall(old, text)) if regex else text.count(old)
    if found == count:
        print(f"    applied   {what} ({count} site(s))")
        return re.sub(old, new.replace("\\", "\\\\"), text) if regex else text.replace(old, new)
    assert found == 0 and text.count(new) == count, f"{what}: expected {count} site(s), found {found}"
    print(f"    present   {what}")
    return text


def signed_remainder(text):
    """1. A sensitised cell in state 5, 7 or 8 with no excited arrow pointing at it.

    input() then returns 0 and the port computed (0 - 9 + k) % 4 in u32, which wraps. Golly's C++
    does it in int, where the remainder is negative: 5 -> 7, 7 -> 15, 8 -> 8. The table has the
    C++ results, so they are returned explicitly."""
    text = patch(
        text,
        "{return ( (quiesce(input(n,s,e,w)) - 9u+3u) % 4u )+9u;}",
        "{ if(input(n,s,e,w)==0u) {return 7u;}	// nothing points at us: (0-9+3)%4 is -2 in C++\n"
        "\t\t\t  return ( (quiesce(input(n,s,e,w)) - 9u+3u) % 4u )+9u; }",
        1,
        "state 5 without input -> 7",
    )
    text = patch(
        text,
        "{return ( (quiesce(input(n,s,e,w)) - 9u+3u) % 4u )+17u;}",
        "{ if(input(n,s,e,w)==0u) {return 15u;}	// nothing points at us: (0-9+3)%4 is -2 in C++\n"
        "\t\t\t  return ( (quiesce(input(n,s,e,w)) - 9u+3u) % 4u )+17u; }",
        1,
        "state 7 without input -> 15",
    )
    text = patch(
        text,
        "{return 9u+dir(input(n,s,e,w));}",
        "{ if(input(n,s,e,w)==0u) {return 8u;}	// nothing points at us: (0-9)%4 is -1 in C++\n"
        "\t\t\t  return 9u+dir(input(n,s,e,w)); }",
        1,
        "state 8 without input -> 8",
    )
    return text


def xnor_inputs(text):
    """2. Hutton32b only. The XNOR gate of the table needs ordinary transmission states as its two
    inputs (e in {11,15} and w in {9,13}, or n in {12,16} and s in {10,14}); the port also
    accepted special ones, which hid 576 ordinary crossings behind the XNOR branch."""
    text = patch(text, "    (is_south(n) && is_north(s) && ", "    (is_OTS(n) && is_OTS(s) && is_south(n) && is_north(s) && ", 1, "XNOR vertical inputs are OTS")
    text = patch(text, "    (is_east(w)  && is_west (e) && ", "    (is_OTS(w) && is_OTS(e) && is_east(w)  && is_west (e) && ", 1, "XNOR horizontal inputs are OTS")
    return text


def crossing_revert(text):
    """3. Hutton32b only. Rules <29,12,9,0,13,25>: an active crossing reverts to 25 when one of its
    neighbours is the ground state. The port reverted whenever the neighbourhood was not a valid
    crossing, and did so before testing for an incoming excited STS, which destroys the cell."""
    old = (
        r"if\(!is_crossing\(n,s,e,w\)\)\s*\{\s*\{return 25u;\}[^\n]*\n\s*\}\s*"
        r"else if\(excited_STS_arrow_to_us\(c,n,s,e,w\)\)\s*\{return 0u;\s*\}[^\n]*\n"
    )
    new = (
        "if(excited_STS_arrow_to_us(c,n,s,e,w))\n"
        "\t\t\t{return 0u;	}// we get destroyed by the incoming excited STS\n"
        "\t\telse if(n==0u || s==0u || e==0u || w==0u)\n"
        "\t\t\t{return 25u;}	// no longer a crossing: a neighbour is the ground state\n"
    )
    return patch(text, old, new, 1, "crossing reverts next to ground, after the STS test", regex=True)


def modern_wgsl(text):
    """Current WGSL (naga 0.20+) rejects two things naga 0.10 accepted. No change in meaning, and
    naga 0.10 accepts the new spelling too."""
    a = "all_inputs_on(n,s,e,w) || xnor_crossing(n,s,e,w) && xnor_output(n,s,e,w)"
    text = patch(text, a, "all_inputs_on(n,s,e,w) || (xnor_crossing(n,s,e,w) && xnor_output(n,s,e,w))", 3, "parenthesise a || b && c")
    text = patch(text, "let output = output(c,n,s,e,w);", "let out_state = output(c,n,s,e,w);", 1, "local no longer shadows output()")
    text = patch(text, "is_sensitized(output) &&", "is_sensitized(out_state) &&", 1, "  use of the renamed local")
    text = patch(text, "is_excited(c) && output==0u", "is_excited(c) && out_state==0u", 1, "  use of the renamed local")
    return text


NOTE = (
    "// Equal to the canonical {rule} on all 2^25 neighbourhoods:\n"
    "// {url}\n"
    "// (checked on the GPU with `ca-research lut --rule {name} --against table:<file>` in research/; paper.md, section 10).\n"
)


def main():
    for name, rule, url, steps in [
        ("hutton32", "Hutton32", "http://gollygang.github.io/ruletablerepository/downloads/Hutton32.table", [signed_remainder]),
        ("hutton32b", "Hutton32b", "https://github.com/FredericaBernkastel/Hutton32-Computer/blob/master/doc/Hutton32b.rule",
         [signed_remainder, xnor_inputs, crossing_revert, modern_wgsl]),
    ]:
        path = RULES / f"{name}.rule.wgsl"
        print(f"{path}:")
        with io.open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        for step in steps:
            text = step(text)
        note = NOTE.format(rule=rule, url=url, name=name)
        if not text.startswith(note):
            text = note + text
        with io.open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)


if __name__ == "__main__":
    main()
