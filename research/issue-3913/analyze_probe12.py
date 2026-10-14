"""Summarize probe12_splitRotary from the iPhone harness report (or an attachment)."""
import collections, glob, json, os, statistics as st, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/Users/patricedery/Coding projects/iRT/build/device-harness/harness-report.json"
data = json.load(open(path))
rows = data["scenarios"]["probe12_splitRotary"]["results"] if "scenarios" in data else data["results"]
by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    by[r["arm"]][r["prefill"]].append(r)

print(f"{len(rows)} runs\n")
for arm, groups in by.items():
    first = next(iter(groups.values()))[0]
    print(f"== {arm}: load {first['loadSeconds']} s, footprint after load {first['loadedFootprintGB']} GB")
    for p in sorted(groups):
        g = groups[p]
        print(f"   {p:>6} tokens: {sum(r['exact'] for r in g)}/{len(g)} exact   prefill {st.median(r['prefillTPS'] for r in g):>6.0f} tok/s   decode {st.median(r['decodeTPS'] for r in g):>5.1f} tok/s")

def text(arm):
    return {(r["sentences"], r["seed"]): r["output"] for r in rows if r["arm"] == arm}
for a, b in [("split-fp16", "original-fp32"), ("split-fp32", "original-fp32"), ("split-fp16-mtp", "original-fp32-mtp")]:
    ta, tb = text(a), text(b)
    common = [k for k in ta if k in tb]
    if common:
        print(f"\n{a} text identical to {b}: {sum(ta[k] == tb[k] for k in common)}/{len(common)}")
print(f"\nfree disk at start: {rows[0]['freeDiskGB']:.1f} GB")
