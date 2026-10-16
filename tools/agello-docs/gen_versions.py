#!/usr/bin/env python3
"""generated/version-matrix.md: versions of everything the reference documents, and what Agello ships."""
import glob
import json
import os
import re
import subprocess

import common

AGELLO = "/Users/patricedery/Coding projects/iRT"


def grab(text, pattern, default="?", flags=re.M):
    m = re.search(pattern, text or "", flags)
    return m.group(1) if m else default


def workspace_repo(ws, name):
    """The commit or URL of an http_archive/local_repository named `name` in a WORKSPACE."""
    m = re.search(r'name = "' + re.escape(name) + r'",(.*?)\n\)', ws, re.S)
    if not m:
        return None
    body = m.group(1)
    for pat in (r'strip_prefix = "[^"]*?-([0-9a-f]{7,40})"', r'commit = "([0-9a-f]{7,40})"',
                r'archive/([0-9a-f]{7,40})\.', r'strip_prefix = "[^"]*" \+ (\w+)', r'urls? = \[?\s*"([^"]+)"', r'path = "([^"]+)"'):
        x = re.search(pat, body)
        if x:
            value = x.group(1)
            v = re.search(rf'^{value} = "([^"]+)"', ws, re.M) if re.fullmatch(r"\w+", value) and not re.fullmatch(r"[0-9a-f]+", value) else None
            return v.group(1) if v else value
    return "(declared)"


def main():
    lm, shipped = "litert-lm", "litert-lm-shipped"
    ws = common.show(lm, "WORKSPACE")
    rows = [
        ("LiteRT-LM (documented)", f"`agello-next` `{common.repo(lm)['pin'][:10]}`",
         f"version.bzl {grab(common.show(lm, 'version.bzl'), r'^VERSION = \"([^\"]+)\"')}, "
         f"C API {grab(common.show(lm, 'version.bzl'), r'^C_API_VERSION = \"([^\"]+)\"')}; "
         f"upstream base {common.repo(lm)['base_label']} `{common.repo(lm)['base'][:10]}`"),
        ("LiteRT-LM (Agello ships)", f"`agello` `{common.repo(shipped)['pin'][:10]}`",
         f"version.bzl {grab(common.show(shipped, 'version.bzl'), r'^VERSION = \"([^\"]+)\"')}, "
         f"C API {grab(common.show(shipped, 'version.bzl'), r'^C_API_VERSION = \"([^\"]+)\"')}; "
         "Swift wrapper only, with Google's prebuilt binary"),
        ("LiteRT", f"`{common.repo('litert')['pin'][:10]}`",
         f"WORKSPACE `AGELLO_LITERT_COMMIT`; upstream base `{common.repo('litert')['base'][:10]}` ({common.repo('litert')['base_label']})"),
        ("ML Drift", f"`{common.repo('ml-drift')['pin'][:10]}`",
         f"WORKSPACE `ML_DRIFT_COMMIT`; upstream base `{common.repo('ml-drift')['base'][:10]}` ({common.repo('ml-drift')['base_label']})"),
        (".litertlm container (writer)", grab(common.show(lm, "schema/core/litertlm_header.h"), r"LITERTLM_MAJOR_VERSION = (\d+)")
         + "." + grab(common.show(lm, "schema/core/litertlm_header.h"), r"LITERTLM_MINOR_VERSION = (\d+)")
         + "." + grab(common.show(lm, "schema/core/litertlm_header.h"), r"LITERTLM_PATCH_VERSION = (\d+)"),
         "`schema/core/litertlm_header.h`"),
        ("Bazel", (common.show(lm, ".bazelversion") or "?").strip(), "`.bazelversion`"),
    ]
    for name in ("org_tensorflow", "XNNPACK", "com_google_absl", "sentencepiece", "FP16"):
        v = workspace_repo(ws, name)
        if v:
            rows.append((f"WORKSPACE `{name}`", f"`{v[:40]}`", "agello-next WORKSPACE"))
    minos = re.findall(r'"(\w+)":\s*"([\d.]+)"', grab(common.show(lm, "agello/BUILD"), r"minimum_os_versions = \{(.*?)\}", "", re.S))
    if minos:
        rows.append(("Our xcframework minimum OS", ", ".join(f"{p} {v}" for p, v in minos), "`agello/BUILD`"))

    info = os.path.join(common.repo(lm)["path"], "dist", "BUILD-INFO.txt")
    build = open(info).read() if os.path.exists(info) else ""
    if build:
        rows.append(("Our built framework (local dist/)", f"build id `{grab(build, r'Build id: (\w+)')}`",
                     f"LiteRT-LM `{grab(build, r'LiteRT-LM: (\w+)')[:10]}`, LiteRT `{grab(build, r'^LiteRT: (\w+)')[:10]}`, "
                     f"ML Drift `{grab(build, r'ML Drift: (\w+)')[:10]}`, built {grab(build, r'Built: (\S+)')}"))

    pkg = subprocess.run(["git", "-C", AGELLO, "show", "HEAD:Vendor/LiteRT-LM/Package.swift"], capture_output=True, text=True).stdout
    urls = re.findall(r'url:\s*"([^"]+)"', pkg)
    sums = re.findall(r'checksum:\s*"([0-9a-f]+)"', pkg)
    for u, c in zip(urls, sums):
        rows.append(("Agello's committed binary target", f"`{os.path.basename(u)}`", f"{u} (checksum `{c[:16]}…`)"))

    for key, m in common.PINS["models"].items():
        j = os.path.join(common.GEN_DIR, "models", f"{key}.json")
        ver = json.load(open(j))["header"]["version"] if os.path.exists(j) else "?"
        rows.append((f"Model `{key}`", f"container {ver}", f"{m['source']}; sha256 `{m['sha256'][:16]}…`"))
    for key, m in common.PINS.get("remote_models", {}).items():
        j = os.path.join(common.GEN_DIR, "models", f"{key}.json")
        info_j = json.load(open(j)) if os.path.exists(j) else {}
        ver = info_j.get("header", {}).get("version", "?")
        sha = info_j.get("sha256")
        rows.append((f"Model `{key}`", f"container {ver}", m["source"] + (f"; sha256 `{sha[:16]}…`" if sha else "")))

    md = [common.gen_header("gen_versions.py", [lm, shipped, "litert", "ml-drift"]), "# Version matrix\n\n",
          "| Component | Version or pin | Where it comes from |\n| --- | --- | --- |\n"]
    md += [f"| {a} | {b} | {c} |\n" for a, b, c in rows]
    print(common.write_gen("version-matrix.md", "".join(md)))


if __name__ == "__main__":
    main()
