#!/usr/bin/env python3
"""generated/c-api-inventory.md: every declaration in LiteRT-LM's public C headers, from clang's AST.

Reads the agello-next worktree, which must be checked out at the pin. One section per header:
functions (signature, comment, permalink), enums (constants and values), typedefs and opaque types.
"""
import json
import os
import subprocess
import sys

import common

KEY = "litert-lm"
HEADERS = ["c/engine.h", "c/conversation.h", "c/embedding_engine.h", "c/model_info.h",
           "c/experimental.h", "c/error_reporter.h", "c/telemetry.h", "c/build_info.h"]


def comment_text(node):
    out = []

    def walk(n):
        if n.get("kind") == "TextComment" and n.get("text", "").strip():
            out.append(n["text"].strip())
        for c in n.get("inner", []):
            walk(c)
    for c in node.get("inner", []):
        if c.get("kind") == "FullComment":
            walk(c)
    return " ".join(out)


def ast(path):
    root = common.repo(KEY)["path"]
    r = subprocess.run(["clang", "-Xclang", "-ast-dump=json", "-fsyntax-only", "-fparse-all-comments",
                        "-x", "c", "-I", root, os.path.join(root, path)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"clang failed on {path}: {r.stderr[:400]}")
    return json.loads(r.stdout)


def decls(path):
    """Top-level declarations located in `path` (clang omits `file` when it repeats the previous one)."""
    current, out = None, []
    for n in ast(path).get("inner", []):
        loc = n.get("loc", {})
        if "file" in loc:
            current = loc["file"]
        elif "expansionLoc" in loc and "file" in loc["expansionLoc"]:
            current = loc["expansionLoc"]["file"]
        if current and current.endswith(path) and not n.get("isImplicit"):
            line = loc.get("line") or loc.get("expansionLoc", {}).get("line")
            out.append((n, line))
    return out


def main():
    root = common.repo(KEY)["path"]
    head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != common.repo(KEY)["pin"]:
        sys.exit(f"{root} is at {head[:10]}, not the pin {common.repo(KEY)['pin'][:10]}")
    dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "c/"], capture_output=True, text=True).stdout
    if dirty.strip():
        sys.exit(f"uncommitted changes in c/: {dirty}")

    md = [common.gen_header("gen_c_api.py", [KEY]), "# LiteRT-LM C API inventory\n\n"]
    version = [l for l in open(os.path.join(root, "c/engine.h")) if "LITERT_LM_C_API_VERSION" in l and "#define" in l]
    md.append("C API version macros (`c/engine.h`): " + ", ".join(f"`{' '.join(l.split()[1:3])}`" for l in version) + "\n\n")
    totals = {}
    for h in HEADERS:
        items = decls(h)
        fns = [(n, l) for n, l in items if n["kind"] == "FunctionDecl"]
        enums = [(n, l) for n, l in items if n["kind"] == "EnumDecl"]
        types = [(n, l) for n, l in items if n["kind"] in ("TypedefDecl", "RecordDecl") and n.get("name")]
        totals[h] = len(fns)
        md.append(f"## `{h}`\n\n{len(fns)} functions, {len(enums)} enums, {len(types)} types. "
                  f"Source: {common.md_link(KEY, h)}\n\n")
        if types:
            md.append("**Types:** " + ", ".join(
                f"[`{n['name']}`]({common.permalink(KEY, h, l)})" for n, l in types) + "\n\n")
        for n, l in enums:
            consts = [c for c in n.get("inner", []) if c["kind"] == "EnumConstantDecl"]
            vals = []
            for c in consts:
                v = next((x.get("value") for x in c.get("inner", []) if "value" in x), None)
                vals.append(f"`{c['name']}`" + (f" = {v}" if v is not None else ""))
            md.append(f"**enum {n.get('name', '(anonymous)')}** ([line {l}]({common.permalink(KEY, h, l)})): "
                      + ", ".join(vals) + "\n\n")
        if fns:
            md.append("| Function | Signature | Comment |\n| --- | --- | --- |\n")
            for n, l in fns:
                params = ", ".join(f"{p['type']['qualType']} {p.get('name', '')}".strip()
                                   for p in n.get("inner", []) if p["kind"] == "ParmVarDecl")
                ret = n["type"]["qualType"].split("(")[0].strip()
                doc = comment_text(n).replace("|", "\\|")
                md.append(f"| [`{n['name']}`]({common.permalink(KEY, h, l)}) | `{ret} ({params})` | {doc} |\n")
            md.append("\n")
    md.append("## Totals\n\n" + "\n".join(f"- `{h}`: {c} functions" for h, c in totals.items())
              + f"\n- all: {sum(totals.values())} functions\n")
    print(common.write_gen("c-api-inventory.md", "".join(md)))


if __name__ == "__main__":
    main()
