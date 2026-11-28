"""Tables of Section VI-I from the logs of `rule-circuits` and `rule-tables`.

usage (from research/): python tools/rule_circuits.py        prints the tables as Markdown
"""
import re
import sys

from ghl_logs import Logs

RULE = re.compile(
    r"^rule (?P<name>\S+) \| (?P<states>\d+) states \((?P<bits>\d+) bits\), (?P<nbhd>\w+), symmetries (?P<sym>\w+)"
    r" \| (?P<lines>\d+) transitions, (?P<cases>\d+) with symmetries, (?P<kept>\d+) after simplification"
    r" \| built: (?P<built>[^|]+) nodes \| (?P<mode>optimised|not optimised): (?P<nodes>\d+) nodes \((?P<style>\w+)\)"
    r" \| (?P<gates>\d+) gates \(AND (?P<and_>\d+), OR (?P<or_>\d+), XOR (?P<xor>\d+), NOT (?P<not_>\d+)\), depth (?P<depth>\d+)"
    r" \| checked on (?P<checked>\d+) neighbourhoods \((?P<how>all|a sample)\)")
NETLIST = re.compile(
    r"^netlist (?P<file>\S+) \| rule (?P<name>\S+), (?P<states>\d+) states \((?P<bits>\d+) bits\), (?P<nbhd>\w+)"
    r" \| (?P<nodes>\d+) nodes \| (?P<gates>\d+) gates \(AND (?P<and_>\d+), OR (?P<or_>\d+), XOR (?P<xor>\d+), NOT (?P<not_>\d+)\), depth (?P<depth>\d+)"
    r" \| equal to the table on (?P<checked>\d+) neighbourhoods \((?P<how>all|a sample)\)")
INTS = ("states", "bits", "lines", "cases", "kept", "nodes", "gates", "and_", "or_", "xor", "not_", "depth", "checked")


def parse(text):
    """The log -> records of `rule` and `netlist` lines, each with its `lut` and `golly` lines."""
    section, out = "", []
    for line in text.splitlines():
        if line.startswith("=== "):
            section = line[4:]
            continue
        m = RULE.match(line) or NETLIST.match(line)
        if m:
            r = m.groupdict()
            r["section"] = section
            if "built" in r:
                r["built"] = dict((k, int(v)) for k, v in (p.strip().split(" ") for p in r["built"].split(",")))
            for k in INTS:
                if k in r:
                    r[k] = int(r[k])
            out.append(r)
        elif line.startswith("lut ") and out:
            m = re.search(r"(\d+) entries, (\d+) differ", line)
            out[-1]["lut"] = (int(m.group(1)), int(m.group(2)))
        elif line.startswith("golly: ") and out:
            out[-1]["golly"] = line[7:]
        elif line.startswith("error"):
            out.append({"error": line, "section": section})
    return out


def num(n):
    return f"{n:,}"


def pow_of(n):
    """n as a power where it is one: 33554432 -> $2^{25}$."""
    if n & (n - 1) == 0:
        return f"$2^{{{n.bit_length() - 1}}}$"
    return num(n)


def soup(r):
    g = r.get("golly", "")
    if "agrees" in g:
        return "agrees"
    if g.startswith("skipped"):
        return "not run"
    return g or "–"


WHAT = {
    "Hutton32b.netlist": "compiled from [`Hutton32b.rule`](research/rules/Hutton32b.rule)",
    "Hutton32.netlist": "compiled from [`Hutton32.table`](research/rules/Hutton32.table)",
    "Hutton32b-from367.netlist": "the optimiser started from the 367-gate circuit",
    "circuit.netlist": "the 367-gate circuit of Section IV-C",
}


def table_circuits(rs):
    print("| Circuit | Rule | Nodes | Gates | AND, OR, XOR, NOT | Depth |")
    print("|---|---|---|---|---|---|")
    for r in rs:
        assert r["how"] == "all" and r["checked"] == 1 << 25 and r["lut"] == (1 << 25, 0) and soup(r) == "agrees", r
        print(f"| {WHAT[r['file']]} | {r['name']} | {r['nodes']} | **{r['gates']}** | {r['and_']}, {r['or_']}, {r['xor']}, {r['not_']} | {r['depth']} |")


def table_golly(rs):
    print("| Rule | States | Neighbourhood, symmetries | Transitions: written, with symmetries | Nodes as built | Compared on | Soup against Golly |")
    print("|---|---|---|---|---|---|---|")
    for r in rs:
        if "error" in r:
            m = re.match(r"error: (\S+): more than (\d+) transitions after expanding symmetries (\w+)", r["error"])
            assert m, r["error"]
            print(f"| {m.group(1)} | | {m.group(3)} | more than {int(m.group(2)):,} with symmetries | not compiled | | |")
            continue
        how = "all " + pow_of(r["checked"]) if r["how"] == "all" else num(r["checked"]) + ", a sample"
        print(f"| {r['name']} | {r['states']} | {r['nbhd']}, {r['sym']} | {num(r['lines'])}, {num(r['cases'])} | {num(r['nodes'])} | {how} | {soup(r)} |")


def main():
    logs = Logs()
    rs = parse("\n".join(logs.lines("rule-circuits.txt") + logs.lines("rule-tables.txt")))
    errors = [r for r in rs if "error" in r]
    circuits = [r for r in rs if "file" in r]
    built = [r for r in rs if "built" in r and "two tables" in r["section"]]
    golly = [r for r in rs if "Golly" in r["section"]]
    print("TABLE compiled\n")
    table_circuits(circuits)
    print()
    for r in built:
        print(f"<!-- as built, {r['name']}: {r['lines']} transitions, {r['kept']} kept; {r['built']}; lut {r.get('lut')}; checked {r['checked']} ({r['how']}) -->")
    print("\nTABLE gollyrules\n")
    table_golly(golly)
    ok = [r for r in golly if "error" not in r]
    print(f"\n<!-- {len(ok)} rules; compared on all neighbourhoods: {sum(r['how'] == 'all' for r in ok)}; on a sample: {[r['name'] for r in ok if r['how'] != 'all']};"
          f" soups agree: {sum(soup(r) == 'agrees' for r in ok)}; no soup: {[(r['name'], r.get('golly')) for r in ok if soup(r) != 'agrees']}; errors: {errors} -->")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
