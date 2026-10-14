#!/usr/bin/env python3
"""Explain agello-bench records with the engine's telemetry. Standard library only.

Usage: python3 telemetry.py results/run.jsonl [more.jsonl ...] [--label ARM] [--case ID]
                            [--weights-gb 2.198]
       python3 telemetry.py <Agello build/…/harness-report.json>   (each scenario is a case)
       python3 telemetry.py results/run.jsonl.load-telemetry.json   (an engine load: delegation,
                                                                     caches, anomalies)
       python3 telemetry.py --graph DIR   (GPU graphs written with AGELLO_TELEMETRY_GRAPH_DIR=DIR)

Needs records from our LiteRT-LM builds (they carry a `telemetry` snapshot per case); run them
with `--telemetry 2` or higher for span timings and GPU time. For each arm and case (warm-ups
left out, repetitions combined), it prints:
  - where each generated token's time goes: the whole step, then its parts on the host (model
    run enqueue, embedding lookup, mask and parameter fills, sampling, readback, detokenizing,
    the stream callback) and on the GPU, with GPU busy time against step time;
  - prefill: chunks, run time, GPU time;
  - effective weight bandwidth (GB of weights read per decode run divided by GPU time);
  - counters that flag waste or silent fallbacks (CPU sampler, KV copies, context switches,
    per-step allocations, host locks), MTP acceptance;
  - host copies between GPU tensors and host memory (bytes per generated token);
  - delegation coverage and program and weight cache use (in load snapshots);
  - anomalies, and LiteRT's own trace sites ranked by time.
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

# Spans normalized per generated token (decode path).
DECODE_SPANS = [
    ("decode.step", "whole step (sample → detokenize)"),
    ("run.decode", "  host: model run (enqueue)"),
    ("embedding_lookup", "  host: embedding lookups (CPU models)"),
    ("mask_fill", "  host: attention mask fill"),
    ("param_fill", "  host: KV parameter fill"),
    ("sample", "  host: sampling"),
    ("logits_readback", "  host: token readback (waits for the GPU)"),
    ("host.lock_read", "  host: GPU → host buffer copies"),
    ("host.unlock_write", "  host: host → GPU buffer copies"),
    ("constraint", "  host: constraint mask"),
    ("detokenize", "  host: detokenize"),
    ("stream_callback", "  host: stream callback"),
    ("run.verify", "  host: MTP verify runs"),
    ("run.drafter", "  host: MTP drafter runs"),
    ("gpu.wait.decode", "  host: explicit GPU waits"),
    ("gpu.decode", "  GPU: decode runs"),
    ("gpu.verify", "  GPU: MTP verify runs"),
    ("gpu.drafter", "  GPU: MTP drafter runs"),
]
PREFILL_SPANS = [
    ("turn.prefill", "prefill turn"),
    ("template", "  template render"),
    ("tokenize", "  tokenize"),
    ("prefill.chunk", "  chunks"),
    ("prefill.embedding_lookup", "    embedding lookups (CPU models)"),
    ("prefill.mask_fill", "    attention mask fill"),
    ("prefill.param_fill", "    KV parameter fill"),
    ("run.prefill", "  model runs (host)"),
    ("gpu.wait.prefill", "  explicit GPU waits"),
    ("gpu.prefill", "  GPU"),
]
WATCH_COUNTERS = [
    "runs_prefill", "runs_decode", "runs_verify", "runs_drafter", "command_buffers", "gpu_waits",
    "sampler_gpu", "sampler_cpu", "logit_masks", "embedding_lookups", "mask_fills",
    "token_history_copies", "decode_allocations", "kv_copies", "context_switches",
    "host_locks", "host_lock_copies", "host_lock_copy_bytes", "host_unlock_copies",
    "host_unlock_copy_bytes", "host_shadow_allocs", "run_option_parses", "tasks_scheduled",
    "stop_eos", "stop_max_tokens", "stop_kv_full", "stop_cancelled", "errors",
]
LOAD_COUNTERS = [
    "compiled_models", "graph_nodes", "graph_nodes_gpu", "graph_nodes_cpu_delegate",
    "graph_nodes_undelegated", "gpu_partitions", "program_cache_hits", "program_cache_misses",
    "weight_cache_hits", "weight_cache_misses",
]


def load(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            text = f.read()
        if path.endswith(".json"):
            report = json.loads(text)
            if isinstance(report, dict) and "scenarios" in report:
                rows += agello_rows(report, path)
                continue
            if isinstance(report, dict) and "counters" in report:
                # A bare snapshot, such as agello-bench's <out>.load-telemetry.json.
                rows.append({"label": "snapshot", "id": os.path.basename(path), "telemetry": report,
                             "warmup": False, "prefill": 0, "decode": 0, "prefillTPS": 0,
                             "decodeTPS": 0})
                continue
        rows += [json.loads(line) for line in text.splitlines() if line.strip()]
    return [r for r in rows if not r.get("warmup") and r.get("telemetry")]


def agello_rows(report, path):
    """Agello's harness report: one pseudo-record per scenario that carries telemetry."""
    rows = []
    for name, entry in sorted(report["scenarios"].items()):
        t = entry.get("telemetry")
        if not t:
            continue
        c = t.get("counters", {})
        rows.append({
            "label": "agello", "id": name, "telemetry": t, "warmup": False,
            "prefill": c.get("tokens_prefilled", 0), "decode": c.get("tokens_decoded", 0),
            "prefillTPS": entry.get("prefillTokPerSec", 0) or 0,
            "decodeTPS": entry.get("decodeTokPerSec", 0) or 0,
        })
    return rows


def combine(records):
    """Sums counters and span totals over repetitions; keeps the worst max."""
    counters = defaultdict(int)
    spans = defaultdict(lambda: {"count": 0, "total_ms": 0.0, "max_us": 0.0})
    anomalies = defaultdict(int)
    for r in records:
        t = r["telemetry"]
        for k, v in (t.get("cpu") or {}).items():
            counters["cpu." + k] += v
        for k, v in t["counters"].items():
            counters[k] += v
        for k, v in t["spans"].items():
            s = spans[k]
            s["count"] += v["count"]
            s["total_ms"] += v["total_ms"]
            s["max_us"] = max(s["max_us"], v["max_us"])
        for a in t["anomalies"]:
            anomalies[(a["code"], a["message"])] += 1
    return counters, spans, anomalies


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*")
    parser.add_argument("--graph", metavar="DIR", help="summarize GPU graph dumps in DIR")
    parser.add_argument("--label")
    parser.add_argument("--case")
    parser.add_argument("--kernels", type=int, default=15, help="kernels listed per phase (level 4)")
    parser.add_argument("--weights-gb", type=float, default=2.198,
                        help="weights read per decode run (E4B: 2.198 GB)")
    args = parser.parse_args()

    if args.graph:
        graphs(args.graph, args.kernels)
        if not args.files:
            return
    rows = load(args.files)
    if not rows:
        print("No records with telemetry (run our build with --telemetry 2).")
        return
    groups = defaultdict(list)
    for r in rows:
        if (args.label and r["label"] != args.label) or (args.case and r["id"] != args.case):
            continue
        groups[(r["label"], r["id"])].append(r)

    for (label, case), records in groups.items():
        counters, spans, anomalies = combine(records)
        n = len(records)
        decoded = sum(r["decode"] for r in records)
        prefilled = sum(r["prefill"] for r in records)
        level = records[0]["telemetry"]["level"]
        if label == "snapshot":
            print(f"\n== {case} (telemetry level {level})")
        else:
            print(f"\n== {label} / {case}: {n} run(s), {prefilled // n} prompt tokens, {decoded // n} "
                  f"generated; prefill {records[0]['prefillTPS']:.0f} tok/s, decode "
                  f"{records[0]['decodeTPS']:.1f} tok/s (telemetry level {level})")
        if level < 2:
            print("  (level 1: counters only; use --telemetry 2 for timings)")
        if decoded and spans:
            print(f"  per generated token, ms (mean over {decoded} tokens):")
            step = spans.get("decode.step", {}).get("total_ms", 0) / decoded
            for key, title in DECODE_SPANS:
                s = spans.get(key)
                if not s or not s["count"]:
                    continue
                per_token = s["total_ms"] / decoded
                share = f"{100 * per_token / step:5.1f}%" if step and key != "decode.step" else "      "
                print(f"    {title:<42} {per_token:8.3f}  {share}  "
                      f"({s['count']} calls, max {s['max_us'] / 1000:.2f} ms)")
            cpu = counters.get("cpu.user_ms", 0) + counters.get("cpu.system_ms", 0)
            if cpu:
                print(f"    process CPU per generated token: {cpu / decoded:.2f} ms "
                      f"(user {counters.get('cpu.user_ms', 0) / decoded:.2f}, system "
                      f"{counters.get('cpu.system_ms', 0) / decoded:.2f}; includes prefill)")
            gpu = sum(spans.get(k, {}).get("total_ms", 0) for k in ("gpu.decode", "gpu.verify", "gpu.drafter"))
            turn = spans.get("turn.decode", {}).get("total_ms", 0)
            if gpu and turn:
                print(f"    GPU busy during decode turns: {100 * gpu / turn:.0f}% of {turn / n:.0f} ms per run")
            runs = counters.get("runs_decode", 0) + counters.get("runs_verify", 0)
            gpu_main = sum(spans.get(k, {}).get("total_ms", 0) for k in ("gpu.decode", "gpu.verify"))
            if runs and gpu_main:
                print(f"    effective weight bandwidth: {args.weights_gb * runs / (gpu_main / 1000):.0f} GB/s "
                      f"({args.weights_gb} GB per decode or verify run, GPU time only)")
            queue = spans.get("gpu.queue_delay")
            if queue and queue["count"]:
                print(f"    command buffers: {queue['count'] / n:.0f} per run; submit → start "
                      f"(includes waiting behind earlier buffers) mean {queue['total_ms'] / queue['count']:.2f} ms")
        if prefilled and spans.get("turn.prefill"):
            print("  prefill, ms per run:")
            for key, title in PREFILL_SPANS:
                s = spans.get(key)
                if s and s["count"]:
                    print(f"    {title:<42} {s['total_ms'] / n:9.2f}  ({s['count'] // n} per run)")
        for phase, runs_key in (("decode", "runs_decode"), ("verify", "runs_verify"), ("prefill", "runs_prefill")):
            prefix = f"gpu.k.{phase}:"
            kernels = sorted(((k[len(prefix):], v) for k, v in spans.items() if k.startswith(prefix)),
                             key=lambda kv: -kv[1]["total_ms"])
            if not kernels:
                continue
            total = sum(v["total_ms"] for _, v in kernels)
            runs = counters.get(runs_key, 0) or 1
            print(f"  GPU kernels per {phase} run, {total / runs:.2f} ms summed (level 4 times each kernel "
                  f"in its own encoder, which runs slower than normal; read the shares):")
            for name, v in kernels[:args.kernels]:
                print(f"    {name:<46} {v['total_ms'] / runs:8.3f} ms  {100 * v['total_ms'] / total:5.1f}%  "
                      f"({v['count'] / runs:.0f} per run)")
        copied = counters.get("host_lock_copy_bytes", 0) + counters.get("host_unlock_copy_bytes", 0)
        if copied and decoded:
            print(f"  host copies (GPU tensors ↔ host memory, prefill included): {copied / decoded / 1024:.1f} KB "
                  f"per generated token; {counters.get('host_lock_copies', 0) // n} reads and "
                  f"{counters.get('host_unlock_copies', 0) // n} writes per run")
        by_size = sorted(((k, v) for k, v in spans.items() if k.startswith(("host.read ", "host.write "))),
                         key=lambda kv: -kv[1]["count"] * int(kv[0].split(" ")[1]))
        if by_size:
            print("  host copies by buffer size (per run):")
            for k, v in by_size[:args.kernels]:
                size = int(k.split(" ")[1])
                print(f"    {k:<46} ×{v['count'] / n:<6.0f} {v['count'] * size / n / 2**20:8.1f} MB  "
                      f"{v['total_ms'] / n:7.2f} ms")
        if counters.get("compiled_models"):
            total = counters.get("graph_nodes", 0)
            print(f"  delegation: {counters['compiled_models'] // n} compiled models, {total // n} nodes: "
                  f"GPU {counters.get('graph_nodes_gpu', 0) // n} in {counters.get('gpu_partitions', 0) // n} "
                  f"partitions, CPU delegate {counters.get('graph_nodes_cpu_delegate', 0) // n}, "
                  f"undelegated {counters.get('graph_nodes_undelegated', 0) // n}")
        caches = {k: counters[k] // n for k in LOAD_COUNTERS[6:] if counters.get(k)}
        if caches:
            print("  caches: " + ", ".join(f"{k} {v}" for k, v in caches.items()))
        drafted, accepted = counters.get("mtp_drafted", 0), counters.get("mtp_accepted", 0)
        if drafted:
            print(f"  MTP: {counters.get('mtp_rounds', 0) // n} rounds per run, acceptance "
                  f"{100 * accepted / drafted:.0f}% ({accepted}/{drafted})")
        facts = records[0]["telemetry"].get("facts") or {}
        if facts:
            print("  facts: " + ", ".join(f"{k} = {v}" for k, v in facts.items()))
        notable = {k: counters[k] for k in WATCH_COUNTERS if counters.get(k)}
        if notable:
            print("  counters (all runs): " + ", ".join(f"{k} {v}" for k, v in notable.items()))
        if counters.get("sampler_cpu"):
            print("  ! sampling ran on the CPU (full logits readback each step)")
        if anomalies:
            print("  anomalies:")
            for (code, message), count in sorted(anomalies.items(), key=lambda x: -x[1]):
                print(f"    {code} ×{count}: {message}")
        dynamic = sorted(((k, v) for k, v in spans.items()
                          if (" " in k or "::" in k) and not k.startswith(("gpu.k.", "host."))),
                         key=lambda kv: -kv[1]["total_ms"])
        if dynamic:
            print("  LiteRT trace sites, ms per run:")
            for k, v in dynamic[:8]:
                print(f"    {k:<46} {v['total_ms'] / n:9.2f}  ({v['count'] // n} per run)")


def graphs(directory, top):
    """GPU graphs from ML Drift (AGELLO_TELEMETRY_GRAPH_DIR): one JSON file per initialized graph."""
    paths = sorted(glob.glob(os.path.join(directory, "gpu-graph-*.json")),
                   key=lambda p: int(os.path.basename(p).split("-")[2]))
    if not paths:
        print(f"No gpu-graph-*.json in {directory}.")
        return
    for path in paths:
        with open(path) as f:
            graph = json.load(f)
        kernels = graph["kernels"]
        flops = sum(k["flops"] for k in kernels)
        print(f"\n== {os.path.basename(path)}: {len(kernels)} kernels ({graph['origin']}), "
              f"{flops / 1e9:.2f} GFLOP per run (shapes {graph['shape_order']})")
        families = Counter()
        family_flops = Counter()
        for k in kernels:
            family = k["name"].split(" -> ")[0]
            families[family] += 1
            family_flops[family] += k["flops"]
        print("  kernel families by count: " + ", ".join(f"{name} {count}" for name, count in families.most_common(top)))
        if flops:
            print("  by FLOPs:")
            for name, value in family_flops.most_common(top):
                print(f"    {name:<46} {value / 1e9:9.3f} GFLOP  {100 * value / flops:5.1f}%")
        fused = sum(1 for k in kernels if " -> " in k["name"])
        print(f"  kernels that fuse several ops: {fused} of {len(kernels)}; families are named by "
              f"their first op")


if __name__ == "__main__":
    main()
