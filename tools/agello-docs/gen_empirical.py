#!/usr/bin/env python3
"""generated/kernels-by-family.md and generated/engine-run.md, from capture_empirical.sh's output.

  gen_empirical.py <capture dir>
The capture dir holds, per GPU family <f>: graphs-<f>/gpu-graph-*.json (ML Drift's graph dumps,
AGELLO_TELEMETRY_GRAPH_DIR), run-<f>.jsonl (agello-bench record with telemetry) and log-<f>.txt.
"""
import collections
import glob
import json
import os
import re
import sys

import common

FAMILIES = [("native", "Apple M6 (native)"), ("a17pro", "Apple A17 Pro (simulated)"),
            ("m2", "Apple M2 (simulated)"), ("m1", "Apple M1 (simulated)")]


def seq_len(g):
    """Tokens per run, from the first kernels: the per-layer-embedding MUL takes [1, S, 42, 1, 256],
    and prefill/verify graphs start by reshaping their S positions ([S, 1, 1, 1, 1])."""
    for k in g["kernels"][:8]:
        for shape in k.get("inputs") or []:
            if shape and len(shape) == 5 and shape[2] == 42 and shape[4] == 256:
                return shape[1]
    first = (g["kernels"][0].get("inputs") or [[1]])[0] if g["kernels"] else [1]
    return first[0] if first and first[0] > 1 else 1


def label_graphs(graphs):
    """Name each dumped graph by signature from its tokens per run (1 decode, 4 verify,
    128 / 1024 prefill); the drafter is the small graph without the per-layer input."""
    names = {1: "decode", 4: "verify", 128: "prefill_128", 1024: "prefill_1024"}
    out = {}
    for path, g in graphs:
        if len(g["kernels"]) < 500:
            out[path] = "mtp_drafter"
        else:
            out[path] = names.get(seq_len(g), f"graph ({seq_len(g)} tokens)")
    return out


def base(name):
    return name.split(" -> ")[0].strip()


def main():
    cap = sys.argv[1]
    per = {}  # family -> signature -> Counter(base kernel)
    flops = {}
    nodes = {}
    for fam, _ in FAMILIES:
        files = sorted(glob.glob(os.path.join(cap, f"graphs-{fam}", "gpu-graph-*.json")))
        graphs = [(f, json.load(open(f))) for f in files]
        names = label_graphs(graphs)
        per[fam], flops[fam], nodes[fam] = {}, {}, {}
        for f, g in graphs:
            sig = names[f]
            per[fam][sig] = collections.Counter(base(k["name"]) for k in g["kernels"])
            flops[fam][sig] = collections.Counter()
            for k in g["kernels"]:
                flops[fam][sig][base(k["name"])] += k.get("flops", 0)
            nodes[fam][sig] = len(g["kernels"])

    sigs = ["decode", "prefill_1024", "prefill_128", "verify", "mtp_drafter"]
    md = [common.gen_header("gen_empirical.py", ["litert-lm", "litert", "ml-drift"]),
          "# Kernels chosen per Apple GPU family\n\n",
          "Gemma 4 E4B (`e4b-ours`, 8K window, MTP on) on this Mac (Apple M6). The simulated rows set "
          "`AGELLO_SIMULATE_APPLE_GPU`, so ML Drift picks the kernels it would pick on that GPU; the hardware, "
          "and its arithmetic, stay the M6's. A kernel name is the first operation of a fused node "
          "(`fc1x1_int8_weights -> cast -> ...` counts as `fc1x1_int8_weights`).\n\n"
          "| Graph | " + " | ".join(label for _, label in FAMILIES) + " |\n| --- |" + " --- |" * len(FAMILIES) + "\n"]
    for sig in sigs:
        md.append(f"| `{sig}` nodes | " + " | ".join(str(nodes[f].get(sig, "-")) for f, _ in FAMILIES) + " |\n")
    for sig in sigs:
        kernels = sorted({k for f, _ in FAMILIES for k in per[f].get(sig, {})},
                         key=lambda k: -max(flops[f].get(sig, {}).get(k, 0) for f, _ in FAMILIES))
        if not kernels:
            continue
        md.append(f"\n## `{sig}`\n\nNode counts per kernel (sorted by FLOPs); rows that differ between families are in bold.\n\n"
                  "| Kernel | " + " | ".join(label for _, label in FAMILIES) + " | GFLOPs (native) |\n| --- |"
                  + " --- |" * (len(FAMILIES) + 1) + "\n")
        for k in kernels:
            counts = [per[f].get(sig, {}).get(k, 0) for f, _ in FAMILIES]
            name = f"**`{k}`**" if len(set(counts)) > 1 else f"`{k}`"
            md.append(f"| {name} | " + " | ".join(str(c) for c in counts)
                      + f" | {flops['native'].get(sig, {}).get(k, 0) / 1e9:,.2f} |\n")
    print(common.write_gen("kernels-by-family.md", "".join(md)))

    # The native run: telemetry facts, counters, memory, and the engine's log lines (deduplicated).
    run = [json.loads(l) for l in open(os.path.join(cap, "run-native.jsonl"))][-1]
    t = run.get("telemetry") or {}
    md = [common.gen_header("gen_empirical.py", ["litert-lm", "litert", "ml-drift"]), "# One engine run, observed\n\n",
          f"agello-bench `speed` case `{run.get('id')}` on {run.get('machine', {}).get('gpu', 'this Mac')}: "
          f"{run.get('prefill')} prompt tokens, {run.get('decode')} generated, 8K window, MTP on, telemetry level {t.get('level')}, "
          f"model `e4b-ours`. Load {run.get('loadSeconds', 0):.1f} s; prefill {run.get('prefillTPS', 0):,.0f} tok/s; "
          f"decode {run.get('decodeTPS', 0):,.1f} tok/s (one run, for orientation only; benchmark mode).\n\n"]
    for fam, label in FAMILIES:
        r = [json.loads(l) for l in open(os.path.join(cap, f"run-{fam}.jsonl"))][-1]
        facts = (r.get("telemetry") or {}).get("facts", {})
        md.append(f"- {label}: " + "; ".join(f"{k} = {v}" for k, v in sorted(facts.items())) + "\n")
    md.append("\n## Counters (native run)\n\n| Counter | Value |\n| --- | --- |\n")
    for k, v in sorted((t.get("counters") or {}).items()):
        md.append(f"| `{k}` | {v} |\n")
    if t.get("gauges"):
        md.append("\n## Gauges (native run)\n\n| Gauge | Value |\n| --- | --- |\n")
        for k, v in sorted(t["gauges"].items()):
            md.append(f"| `{k}` | {v} |\n")
    if t.get("memory"):
        md.append("\n## Memory (native run)\n\n```\n" + json.dumps(t["memory"], indent=1)[:4000] + "\n```\n")
    if t.get("anomalies"):
        kinds = collections.Counter(a.get("code") for a in t["anomalies"])
        md.append("\n## Anomalies (native run)\n\n" + "\n".join(f"- `{k}` x{n}" for k, n in kinds.most_common()) + "\n")
    lines, seen = [], set()
    for line in open(os.path.join(cap, "log-native.txt")):
        m = re.search(r"\[([\w.]+:\d+)\]\s*(.*)", line)
        if not m:
            continue
        key = (m.group(1), re.sub(r"0x[0-9a-f]+|\d+", "N", m.group(2)))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"[{m.group(1)}] {m.group(2).strip()[:220]}")
    md.append("\n## Engine log, first occurrence of each message (native run)\n\n```\n" + "\n".join(lines) + "\n```\n")
    print(common.write_gen("engine-run.md", "".join(md)))


if __name__ == "__main__":
    main()
