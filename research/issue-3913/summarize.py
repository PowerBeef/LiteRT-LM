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
