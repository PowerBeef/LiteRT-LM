#!/usr/bin/env python3
"""Writes agello/THIRD_PARTY_NOTICES.md: the licenses of everything linked into //agello:CLiteRTLM.

Usage (from the repository root, after a build with the same flags):
  python3 agello/third_party_notices.py > agello/THIRD_PARTY_NOTICES.md

The components are the external repositories among the inputs of the framework's link actions
(bazel aquery), so build-time tools (Python, Rust tooling, code generators) aren't listed. Each
repository's license comes from its Cargo.toml `license` field (Rust crates) or the table below
(C/C++), and its text from the repository's LICENSE/COPYING files, or the stated file. Identical
texts are printed once with every component that uses them.
"""
import glob
import json
import os
import re
import subprocess
import sys

# Repositories that are toolchains or build machinery, not code in the binary.
TOOLING = {"bazel_tools", "build_bazel_apple_support", "rules_rust", "platforms"}

# C/C++ repositories: (name, SPDX license, file with the text, relative to the repository).
KNOWN = {
    "litert": ("LiteRT (PowerBeef/LiteRT, branch agello-next)", "Apache-2.0", "LICENSE"),
    "ml_drift": ("ML Drift (PowerBeef/ml-drift, branch agello-next)", "Apache-2.0", "LICENSE"),
    "agello_telemetry": ("Agello telemetry (this repository, agello/telemetry)", "Apache-2.0", None),
    "com_google_absl": ("Abseil C++", "Apache-2.0", "LICENSE"),
    "com_google_protobuf": ("Protocol Buffers", "BSD-3-Clause", "LICENSE"),
    "XNNPACK": ("XNNPACK", "BSD-3-Clause", "LICENSE"),
    "ruy": ("ruy", "Apache-2.0", "LICENSE"),
    "KleidiAI": ("KleidiAI", "Apache-2.0", "LICENSES/Apache-2.0.txt"),
    "flatbuffers": ("FlatBuffers", "Apache-2.0", "LICENSE"),
    "zlib": ("zlib", "Zlib", "LICENSE"),
    "minizip": ("minizip (zlib contrib)", "Zlib", None),
    "sentencepiece": ("SentencePiece", "Apache-2.0", "LICENSE"),
    "tokenizers_cpp": ("tokenizers-cpp", "Apache-2.0", "LICENSE"),
    "stb": ("stb", "MIT OR Unlicense", "LICENSE"),
    "pthreadpool": ("pthreadpool", "BSD-2-Clause", "LICENSE"),
    "miniaudio": ("miniaudio", "MIT-0 OR Unlicense", "LICENSE"),
    "kissfft": ("KISS FFT", "BSD-3-Clause", "COPYING"),
    "fft2d": ("Ooura's FFT (fft2d)", "LicenseRef-Ooura", "readme2d.txt"),
    "farmhash_archive": ("FarmHash", "MIT", "COPYING"),
    "cpuinfo": ("cpuinfo", "BSD-2-Clause", "LICENSE"),
    "com_googlesource_code_re2": ("RE2", "BSD-3-Clause", "LICENSE"),
    "eigen_archive": ("Eigen", "MPL-2.0", "COPYING.MPL2"),
    "gemmlowp": ("gemmlowp", "Apache-2.0", "LICENSE"),
}

LICENSE_FILES = ["LICENSE", "LICENSE.md", "LICENSE.txt", "LICENSE-MIT", "LICENSE-APACHE",
                 "COPYING", "COPYING.txt", "UNLICENSE"]


def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def linked_repositories():
    text = run(["bazelisk", "aquery", "--config=agello", "-c", "opt",
                'mnemonic("ObjcLink|CppLink", deps(//agello:CLiteRTLM))', "--output=text"])
    return sorted(set(re.findall(r"external/([^/ ]+)/", text)))


def license_text(directory, name):
    candidates = [name] if name else LICENSE_FILES
    for candidate in candidates:
        path = os.path.join(directory, candidate)
        if os.path.isfile(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read().strip()
    return None


def main():
    output_base = run(["bazelisk", "info", "output_base"]).strip()
    components = []  # (title, spdx, text)
    rust_std = False
    for repo in linked_repositories():
        if repo in TOOLING:
            continue
        if repo.startswith("rust_") and repo.endswith("_tools"):
            rust_std = True
            continue
        directory = os.path.join(output_base, "external", repo)
        if repo.startswith("crate_index__"):
            crate = repo[len("crate_index__"):]
            spdx = "unknown"
            cargo = os.path.join(directory, "Cargo.toml")
            if os.path.isfile(cargo):
                match = re.search(r'^license\s*=\s*"([^"]+)"', open(cargo).read(), re.M)
                spdx = match.group(1) if match else spdx
            components.append((f"{crate} (Rust crate)", spdx, license_text(directory, None)))
            continue
        title, spdx, file = KNOWN.get(repo, (repo, "unknown", None))
        components.append((title, spdx, license_text(directory, file)))
    if rust_std:
        components.append(("Rust standard library", "MIT OR Apache-2.0", None))

    out = sys.stdout
    out.write("# Third-party notices: CLiteRTLM.xcframework\n\n")
    out.write("Built from PowerBeef/agello-litert-lm (branch agello-next), a fork of LiteRT-LM\n"
              "(Copyright Google LLC, Apache-2.0), with LiteRT and ML Drift (Apache-2.0) linked\n"
              "statically. It also contains the components below, listed from the framework's link\n"
              "inputs by agello/third_party_notices.py.\n\n")
    out.write("| Component | License |\n| --- | --- |\n")
    for title, spdx, _ in components:
        out.write(f"| {title} | {spdx} |\n")
    out.write("\n## License texts\n\n")
    out.write("Components without a text below are covered by the standard text of their license\n"
              "(Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0; MIT: https://opensource.org/license/mit).\n\n")
    by_text = {}
    for title, spdx, text in components:
        if text:
            by_text.setdefault(text, []).append(title)
    for text, titles in by_text.items():
        out.write("### " + ", ".join(titles) + "\n\n```\n" + text + "\n```\n\n")


if __name__ == "__main__":
    main()
