#!/usr/bin/env python3
"""generated/binaries.md: Google's prebuilt CLiteRTLM (what Agello ships) next to ours, from nm/otool/strings.

  gen_binaries.py [<google CLiteRTLM> [<our CLiteRTLM>]]
Defaults: the ios-arm64 slices of Google's v0.18.0 xcframework (agello-bench's .build-google
artifacts) and our dist/ build.
"""
import collections
import os
import re
import subprocess
import sys

import common

HERE_BENCH = os.path.join(common.HERE, "..", "agello-bench")
GOOGLE = os.path.join(HERE_BENCH, ".build-google/artifacts/litert-lm/CLiteRTLM/CLiteRTLM.xcframework/ios-arm64/CLiteRTLM.framework/CLiteRTLM")
OURS = os.path.join(common.repo("litert-lm")["path"], "dist/CLiteRTLM.xcframework/ios-arm64/CLiteRTLM.framework/CLiteRTLM")

# Components to look for (symbol substrings). Closed parts per docs/agello/AUDIT.md §3; the open ones are for contrast.
MARKERS = [
    ("ML Drift Metal accelerator (open since 2026-10-08)", ["GpuMetalAccelerator", "CreateMlDriftMetalDelegate"]),
    ("TopK Metal sampler (closed)", ["TopKMetalSampler", "LiteRtTopKMetalSampler"]),
    ("Gemma model constraint provider / FST (closed)", ["GemmaModelConstraintProvider"]),
    ("llguidance constraints", ["LlgConstraint", "llg_"]),
    ("GPU artisan executor (closed)", ["GpuArtisan", "gpu_artisan", "LlmGpuArtisan"]),
    ("TENSOR_LLM / Google Tensor", ["TensorLlm", "TENSOR_LLM", "google_tensor"]),
    ("LiteRT compiled-model executor (open)", ["LlmLiteRtCompiledModelExecutor"]),
    ("MTP drafter (open)", ["LlmLiteRtMtpDrafter"]),
    ("XNNPACK", ["xnn_create_runtime", "xnn_initialize"]),
    ("KleidiAI", ["kai_"]),
    ("SentencePiece", ["sentencepiece"]),
    ("HuggingFace tokenizers", ["tokenizers_new_from_str", "HuggingFaceTokenizer"]),
    ("minijinja", ["minijinja"]),
    ("Agello telemetry (ours)", ["agello_telemetry", "litert_lm_telemetry_"]),
    ("Agello build info (ours)", ["litert_lm_get_build_info"]),
]


def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def describe(path):
    exported = run("nm", "-gU", "-j", path).split()
    defined = run("nm", "-U", "-j", path).split()
    all_syms = run("nm", "-j", path).split()
    strings = run("strings", "-a", path)
    linked = [l.strip().split(" (")[0] for l in run("otool", "-L", path).splitlines()[1:]]
    c_api = sorted({s.lstrip("_") for s in exported if s.lstrip("_").startswith("litert_lm_")})
    found = {}
    for name, needles in MARKERS:
        n_def = sum(1 for sym in defined if any(n in sym for n in needles))
        n_str = sum(strings.count(n) for n in needles)
        found[name] = (n_def, n_str)
    return {"bytes": os.path.getsize(path), "exported": len(exported), "symbols": len(all_syms),
            "c_api": c_api, "linked": linked, "markers": found}


def main():
    google = sys.argv[1] if len(sys.argv) > 1 else GOOGLE
    ours = sys.argv[2] if len(sys.argv) > 2 else OURS
    g, o = describe(google), describe(ours)
    md = [common.gen_header("gen_binaries.py", ["litert-lm", "Google's v0.18.0 CLiteRTLM.xcframework"]),
          "# CLiteRTLM binaries: Google's v0.18.0 (shipped) and ours\n\n",
          "iOS arm64 slices. For each component: defined symbols whose (mangled) name contains one of its markers, "
          "then mentions in the binary's strings. Defined symbols mean the code is linked in; a name that appears "
          "only in strings is a reference (a dlopen name, a setting, a log message), not an implementation. Defined "
          "matches can still be open code that only configures or loads a closed part (`GpuArtisanConfig`, the dlopen "
          "wrapper for the Metal sampler): compare the two columns rather than reading one alone.\n\n",
          "| | Google v0.18.0 | Ours |\n| --- | --- | --- |\n",
          f"| Size | {g['bytes'] / 1e6:,.1f} MB | {o['bytes'] / 1e6:,.1f} MB |\n",
          f"| Symbols (all / exported) | {g['symbols']:,} / {g['exported']:,} | {o['symbols']:,} / {o['exported']:,} |\n",
          f"| Exported `litert_lm_*` C functions | {len(g['c_api'])} | {len(o['c_api'])} |\n"]
    for name, _ in MARKERS:
        gd, gs = g["markers"][name]
        od, os_ = o["markers"][name]
        md.append(f"| {name} (defined / in strings) | {gd} / {gs} | {od} / {os_} |\n")
    md.append("\n## Linked libraries and frameworks\n\n| Google v0.18.0 | Ours |\n| --- | --- |\n")
    gl, ol = g["linked"], o["linked"]
    for i in range(max(len(gl), len(ol))):
        md.append(f"| {gl[i] if i < len(gl) else ''} | {ol[i] if i < len(ol) else ''} |\n")
    only_g = sorted(set(g["c_api"]) - set(o["c_api"]))
    only_o = sorted(set(o["c_api"]) - set(g["c_api"]))
    md.append(f"\n## C functions exported by one binary only\n\nGoogle's v0.18.0 only ({len(only_g)}): "
              + (", ".join(f"`{x}`" for x in only_g) or "none") + "\n\n"
              f"Ours only ({len(only_o)}): " + (", ".join(f"`{x}`" for x in only_o) or "none") + "\n")
    print(common.write_gen("binaries.md", "".join(md)))


if __name__ == "__main__":
    main()
