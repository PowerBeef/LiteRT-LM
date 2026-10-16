#!/usr/bin/env python3
"""generated/swift-api-inventory.md: the public API of the LiteRTLM Swift module, from a symbol graph.

Usage: gen_swift_api.py <LiteRTLM.symbols.json> [<build log>]
(gen_swift_api.sh builds the symbol graph in a scratch folder and calls this.)
"""
import collections
import json
import os
import sys

import common

KEY = "litert-lm"


def decl(sym):
    return "".join(f["spelling"] for f in sym.get("declarationFragments", [])).strip()


def doc(sym, first_only=False):
    lines = [l["text"] for l in sym.get("docComment", {}).get("lines", [])]
    text = " ".join(l.strip() for l in lines).strip()
    if first_only and text:
        text = text.split(". ")[0].rstrip(".") + "."
    return text.replace("|", "\\|")


def where(sym):
    loc = sym.get("location")
    if not loc:
        return ""
    path = loc["uri"].replace("file://", "")
    root = common.repo(KEY)["path"]
    rel = os.path.relpath(path.replace("%20", " "), root)
    line = loc["position"]["line"] + 1
    return f"[`{rel}:{line}`]({common.permalink(KEY, rel, line)})"


def main():
    graph = json.load(open(sys.argv[1]))
    log = open(sys.argv[2]).read() if len(sys.argv) > 2 else ""
    syms = [s for s in graph["symbols"] if s.get("accessLevel") in ("public", "open")]
    by_type = collections.defaultdict(list)
    tops = {}
    for s in syms:
        path = s["pathComponents"]
        if len(path) == 1:
            tops[path[0]] = s
        else:
            by_type[path[0]].append(s)

    md = [common.gen_header("gen_swift_api.sh", [KEY]), "# LiteRTLM Swift API inventory\n\n",
          f"Module `{graph['module']['name']}`, {len(syms)} public symbols, {len(tops)} top-level declarations. "
          "Built with `swift package dump-symbol-graph --minimum-access-level public` against `dist/CLiteRTLM.xcframework`.\n\n"]
    if "LiteRTLMFoundationModels" in log and "error" in log:
        md.append("> The `LiteRTLMFoundationModels` target (`swift/apple_fm/`) didn't compile against the macOS 27 SDK "
                  "when this was generated, so it isn't listed.\n\n")
    kinds = collections.Counter(tops[t]["kind"]["displayName"] for t in tops)
    def plural(k, n):
        k = k.lower()
        return k if n == 1 else (k + "es" if k.endswith("s") else k + "s")
    md.append("Top level: " + ", ".join(f"{n} {plural(k, n)}" for k, n in kinds.most_common()) + "\n\n")
    for name in sorted(tops, key=lambda n: (tops[n]["kind"]["displayName"], n)):
        t = tops[name]
        md.append(f"## {name} ({t['kind']['displayName'].lower()})\n\n`{decl(t)}`  {where(t)}\n\n")
        if doc(t):
            md.append(doc(t) + "\n\n")
        members = sorted(by_type.get(name, []), key=lambda s: (s["kind"]["displayName"], s["pathComponents"]))
        if members:
            md.append("| Member | Kind | Declaration | Summary |\n| --- | --- | --- | --- |\n")
            for m in members:
                md.append(f"| {where(m) or '`' + '.'.join(m['pathComponents'][1:]) + '`'} "
                          f"| {m['kind']['displayName']} | `{decl(m).replace('|', chr(92) + '|')}` | {doc(m, True)} |\n")
            md.append("\n")
    print(common.write_gen("swift-api-inventory.md", "".join(md)))


if __name__ == "__main__":
    main()
