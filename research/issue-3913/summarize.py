"""Summarize the results in results/ (no dependencies)."""
import glob, json, os

here = os.path.dirname(os.path.abspath(__file__))
load = lambda f: [json.loads(l) for l in open(f)]

print("== Mac (WebGPU delegate), exact replies per arm")
arms = {}
for f in sorted(glob.glob(os.path.join(here, "results/mac/*.jsonl"))):
    cases, model, config = os.path.basename(f)[:-6].split("__")
    rows = load(f)
    arms[(cases, model, config)] = {r["id"]: r for r in rows}
    print(f"  {cases:<10} {model:<16} {config:<9} {sum(r['exact'] for r in rows):>3}/{len(rows):<3}"
          f" prompts {min(r['prefill'] for r in rows)}-{max(r['prefill'] for r in rows)} tokens")

print("\n== Null test: patched vs original text with FP32 activations (Mac)")
same = total = 0
for (cases, model, config), rows in arms.items():
    ref = arms.get((cases, "original", config))
    if model == "original" or config != "gpu-fp32" or not ref:
        continue
    same += sum(rows[k]["output"] == ref[k]["output"] for k in ref)
    total += len(ref)
print(f"  identical text: {same}/{total}")

cross = arms.get(("cross4096", "int-minus2048", "gpu-fp16"))
if cross:
    print("\n== Boundary with the INT32 -2048 shift (Mac): rotary position = real - 2048")
    groups = {}
    for r in cross.values():
        groups.setdefault(r["prefill"], []).append(r)
    for prefill, rows in sorted(groups.items()):
        firsts = sorted(r["firstMismatchPosition"] for r in rows if not r["exact"])
        print(f"  digits at positions {prefill - 1}-{prefill + 46}: {sum(r['exact'] for r in rows)}/{len(rows)} exact"
              + (f"; first errors at {firsts[0]}-{firsts[-1]} (rotary {firsts[0] - 2048}-{firsts[-1] - 2048})" if firsts else ""))

xl_split, xl_ref = arms.get(("xl", "split", "gpu-fp16")), arms.get(("xl", "original", "gpu-fp32"))
if xl_split and xl_ref:
    print("\n== Split patch at 6K-30K tokens (Mac, maxNumTokens 32768)")
    by_len = {}
    for k, r in xl_split.items():
        by_len.setdefault(r["prefill"], []).append(k)
    xl_o16 = arms.get(("xl", "original", "gpu-fp16"), {})
    for prefill, keys in sorted(by_len.items()):
        print(f"  {prefill:>6} tokens: split FP16 {sum(xl_split[k]['exact'] for k in keys)}/{len(keys)},"
              f" original FP32 {sum(xl_ref[k]['exact'] for k in keys)}/{len(keys)},"
              f" original FP16 {sum(xl_o16[k]['exact'] for k in keys) if xl_o16 else '-'}/{len(keys)};"
              f" split FP16 text = original FP32 text: {sum(xl_split[k]['output'] == xl_ref[k]['output'] for k in keys)}/{len(keys)}")

def medians(rows_by_id, key):
    groups = {}
    for r in rows_by_id.values():
        groups.setdefault(r["prefill"], []).append(r[key])
    return {p: sorted(v)[len(v) // 2] for p, v in groups.items()}

speed = {name: {} for name in ("split FP16", "original FP16", "original FP32")}
for (cases, model, config), rows in arms.items():
    if cases not in ("speed", "xl") or "prefillTPS" not in next(iter(rows.values())):
        continue
    name = {("split", "gpu-fp16"): "split FP16", ("original", "gpu-fp16"): "original FP16",
            ("original", "gpu-fp32"): "original FP32"}.get((model, config))
    if name:
        for key in ("prefillTPS", "decodeTPS"):
            speed[name].setdefault(key, {}).update(medians(rows, key))
if all(speed.values()):
    print("\n== Speed (Mac, tokens/s, medians; no speculative decoding)")
    print("  prompt   prefill: split16 orig16 orig32   decode: split16 orig16 orig32")
    for p in sorted(speed["split FP16"]["prefillTPS"]):
        pre = [speed[n]["prefillTPS"].get(p) for n in speed]
        dec = [speed[n]["decodeTPS"].get(p) for n in speed]
        if None in pre + dec:
            continue
        print(f"  {p:>6}            {pre[0]:>6.0f} {pre[1]:>6.0f} {pre[2]:>6.0f}           {dec[0]:>6.1f} {dec[1]:>6.1f} {dec[2]:>6.1f}")

mtp = {(m, c): arms.get(("mtp", m, c)) for m, c in (("split", "gpu-fp16"), ("split-main-only", "gpu-fp16"), ("original", "gpu-fp16"), ("original", "gpu-fp32"))}
if all(mtp.values()):
    print("\n== With speculative decoding (MTP=1, Mac): decode tokens/s, medians (exact copies)")
    print("  prompt   split+drafter   split main-only   original FP16   original FP32")
    dec = {m: medians(rows, "decodeTPS") for m, rows in mtp.items()}
    for p in sorted(dec[("split", "gpu-fp16")]):
        ex = {m: sum(r["exact"] for r in rows.values() if r["prefill"] == p) for m, rows in mtp.items()}
        print(f"  {p:>6}   " + "   ".join(f"{dec[m][p]:>7.1f} ({ex[m]}/3)" for m in mtp))

print("\n== iPhone (Metal accelerator)")
p10 = json.load(open(os.path.join(here, "results/iphone/probe10_position_boundary.json")))
early = {}
for r in p10:
    if r["arm"] == "fp16" and not r["late"]:
        early[(r["sentences"], r["seed"])] = r
below = [r for r in early.values() if r["prefill"] - 1 + 47 < 2048]
above = [r for r in early.values() if r["prefill"] - 1 >= 2048]
crossing = [r for r in early.values() if r not in below and r not in above]
fp32 = [r for r in p10 if r["arm"] == "fp32"]
print(f"  probe 10, FP16: digits all below 2048 {sum(r['exact'] for r in below)}/{len(below)} exact;"
      f" crossing {sum(r['exact'] for r in crossing)}/{len(crossing)};"
      f" all above {sum(r['exact'] for r in above)}/{len(above)}; FP32 {sum(r['exact'] for r in fp32)}/{len(fp32)}")
first = [r["firstMismatchPosition"] for r in crossing + above if not r["exact"]]
print(f"  earliest first error at position {min(first)}")
p11 = json.load(open(os.path.join(here, "results/iphone/probe11_shifted_rotary.json")))
for arm in ("original", "ishift-2048"):
    rows = [r for r in p11 if r["arm"] == arm]
    print(f"  probe 11, FP16, {arm}: {sum(r['exact'] for r in rows)}/{len(rows)} exact")

p12_path = os.path.join(here, "results/iphone/probe12_split_rotary.json")
if os.path.exists(p12_path):
    p12 = json.load(open(p12_path))["results"]
    print("\n== iPhone, split patch (probe 12, maxNumTokens 32768): exact, prefill / decode tok/s (medians)")
    arms12 = {}
    for r in p12:
        arms12.setdefault(r["arm"], {}).setdefault(r["prefill"], []).append(r)
    for arm, groups in arms12.items():
        cells = []
        for p in sorted(groups):
            g = groups[p]
            pre = sorted(r["prefillTPS"] for r in g)[len(g) // 2]
            dec = sorted(r["decodeTPS"] for r in g)[len(g) // 2]
            cells.append(f"{p}: {sum(r['exact'] for r in g)}/{len(g)} {pre:.0f}/{dec:.1f}")
        print(f"  {arm:<18} " + "  ".join(cells))
    text = lambda arm: {(r["sentences"], r["seed"]): r["output"] for r in p12 if r["arm"] == arm}
    for a, b in (("split-fp16", "original-fp32"), ("split-fp32", "original-fp32"), ("split-fp16-mtp", "original-fp32-mtp")):
        ta, tb = text(a), text(b)
        keys = [k for k in ta if k in tb]
        print(f"  {a} text identical to {b}: {sum(ta[k] == tb[k] for k in keys)}/{len(keys)}")
