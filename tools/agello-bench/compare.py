#!/usr/bin/env python3
"""Compare agello-bench results (JSONL) across arms. Standard library only.

Usage: python3 compare.py results/*.jsonl [--base LABEL] [--metrics prefillTPS decodeTPS ttft]

Records are grouped by arm (`label`); warm-up records are left out.
  - Fidelity: exact digit copies per arm, and identical replies against the base arm.
  - Speed: per case, the median and interquartile range of each metric over its
    repetitions, then each arm's change against the base arm with a 95% bootstrap
    confidence interval. The noise band is the base arm against itself: its
    even repetitions against its odd ones (ABAB order gives both halves the same
    conditions). A change inside the noise band isn't a change.
"""
import argparse
import json
import math
import random
import statistics
from collections import defaultdict

BOOTSTRAP = 2000


def load(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if not r.get("warmup")]


def iqr(values):
    if len(values) < 2:
        return 0.0
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[2] - q[0]


def change(new, old):
    """Percent change of the median, with a bootstrap 95% interval."""
    base = statistics.median(old)
    point = 100 * (statistics.median(new) / base - 1)
    rng = random.Random(0)
    draws = sorted(
        100 * (statistics.median(rng.choices(new, k=len(new))) / statistics.median(rng.choices(old, k=len(old))) - 1)
        for _ in range(BOOTSTRAP))
    return point, draws[int(0.025 * BOOTSTRAP)], draws[int(0.975 * BOOTSTRAP) - 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--base", help="the arm others are compared with (default: the first label seen)")
    parser.add_argument("--metrics", nargs="+", default=["prefillTPS", "decodeTPS", "ttft"])
    args = parser.parse_args()

    rows = load(args.files)
    labels = list(dict.fromkeys(r["label"] for r in rows))
    base = args.base or labels[0]
    if base not in labels:
        parser.error(f"no records for --base {base}; arms: {', '.join(labels)}")
    by = defaultdict(list)  # (label, case id) -> rows
    for r in rows:
        by[(r["label"], r["id"])].append(r)
    cases = list(dict.fromkeys(r["id"] for r in rows))

    print("== Arms")
    for label in labels:
        mine = [r for r in rows if r["label"] == label]
        first = mine[0]
        m = first["machine"]
        settings = f"{first['backend']} {first['activation']} kv {first['kv']} mtp {first['mtp']}"
        print(f"  {label}: {len(mine)} records, {first['model']}, {settings}; {m['hardware']} {m['gpu']} "
              f"(Apple{m['gpuAppleFamily']}), runtime {m['runtime'].rsplit('/', 1)[-1]} {m['runtimeSHA256'][:12]}")

    print("\n== Fidelity (greedy decoding: replies should match the base arm's)")
    for label in labels:
        mine = [r for r in rows if r["label"] == label and r["kind"] == "digit-copy"]
        if mine:
            wrong = sorted(r["firstMismatchPosition"] for r in mine if not r["exact"])
            print(f"  {label}: {sum(r['exact'] for r in mine)}/{len(mine)} digit copies exact"
                  + (f"; first wrong digits at positions {wrong[0]}-{wrong[-1]}" if wrong else ""))
    for label in labels:
        if label == base:
            continue
        pairs = [(by[(label, c)][0], by[(base, c)][0]) for c in cases if by[(label, c)] and by[(base, c)]]
        if pairs:
            same = sum(a["output"] == b["output"] for a, b in pairs)
            print(f"  {label} replies identical to {base}: {same}/{len(pairs)} cases")

    for metric in args.metrics:
        print(f"\n== {metric}: median (IQR) per case; change against {base} with its 95% interval")
        print(f"  {'case':<22} {'prefill':>7}  " + "  ".join(f"{label:>24}" for label in labels))
        changes = defaultdict(list)
        for c in cases:
            cells = []
            for label in labels:
                values = [r[metric] for r in by[(label, c)] if r.get(metric) is not None]
                if not values:
                    cells.append(f"{'-':>24}")
                    continue
                cell = f"{statistics.median(values):.1f} ({iqr(values):.1f})"
                old = [r[metric] for r in by[(base, c)]]
                if label != base and old:
                    point, low, high = change(values, old)
                    changes[label].append(point)
                    cell += f" {point:+.1f}% [{low:+.1f},{high:+.1f}]"
                cells.append(f"{cell:>24}")
            prefill = next((r["prefill"] for label in labels for r in by[(label, c)]), 0)
            print(f"  {c:<22} {prefill:>7}  " + "  ".join(cells))
        noise = []
        for c in cases:
            values = [r[metric] for r in by[(base, c)]]  # file order: the order they ran
            if len(values) >= 4:
                noise.append(abs(change(values[0::2], values[1::2])[0]))
        if noise:
            print(f"  noise band ({base} even vs odd repetitions): median {statistics.median(noise):.1f}%, "
                  f"max {max(noise):.1f}%")
        for label, points in changes.items():
            mean = 100 * (math.exp(statistics.fmean(math.log(1 + p / 100) for p in points)) - 1)
            print(f"  {label} vs {base}: geometric mean {mean:+.1f}% over {len(points)} cases")


if __name__ == "__main__":
    main()
