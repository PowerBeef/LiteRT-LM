#!/usr/bin/env python3
"""generated/proto-fields.md: messages, fields and enums of LiteRT-LM's runtime/proto/*.proto at the pin.

A line-based parser (no protoc here): comments directly above a declaration become its description.
"""
import re

import common

KEY = "litert-lm"
FIELD = re.compile(r"^\s*(optional |repeated )?(map<[^>]+>|[\w.]+)\s+(\w+)\s*=\s*(\d+)\s*(\[[^\]]*\])?\s*;\s*(//\s*(.*))?$")
ENUMVAL = re.compile(r"^\s*(\w+)\s*=\s*(-?\d+)\s*(\[[^\]]*\])?\s*;\s*(//\s*(.*))?$")
OPEN = re.compile(r"^\s*(message|enum|oneof)\s+(\w+)\s*\{")


def parse(path, text):
    """Rows: (scope, kind, name, type, number, comment, line)."""
    rows, stack, pending = [], [], []
    for n, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("//"):
            pending.append(s[2:].strip())
            continue
        if not s:
            pending = []
            continue
        m = OPEN.match(line)
        if m:
            kind, name = m.groups()
            scope = ".".join(x[1] for x in stack if x[0] != "oneof")
            rows.append((scope, kind, name, "", "", " ".join(pending), n))
            stack.append((kind, name))
            pending = []
            if s.endswith("}"):
                stack.pop()
            continue
        if s.startswith("}"):
            if stack:
                stack.pop()
            pending = []
            continue
        scope = ".".join(x[1] for x in stack if x[0] != "oneof")
        in_enum = bool(stack) and stack[-1][0] == "enum"
        m = ENUMVAL.match(line) if in_enum else FIELD.match(line)
        if m and in_enum:
            rows.append((scope, "value", m.group(1), "", m.group(2), " ".join(pending + [m.group(5) or ""]).strip(), n))
        elif m:
            label = (m.group(1) or "").strip()
            oneof = stack[-1][1] if stack and stack[-1][0] == "oneof" else ""
            typ = (label + " " if label else "") + m.group(2) + (f" (oneof {oneof})" if oneof else "")
            rows.append((scope, "field", m.group(3), typ, m.group(4), " ".join(pending + [m.group(7) or ""]).strip(), n))
        pending = []
    return rows


def main():
    files = [f for f in common.ls_files(KEY, "runtime/proto") if f.endswith(".proto")]
    md = [common.gen_header("gen_protos.py", [KEY]), "# LiteRT-LM proto messages and fields\n\n",
          "Metadata stored in `.litertlm` files (`LlmMetadata`, `ExecutorMetadata`, `EmbeddingMetadata`, ...) and "
          "settings messages. One section per file.\n\n"]
    for f in files:
        rows = parse(f, common.show(KEY, f))
        md.append(f"## `{f}`\n\nSource: {common.md_link(KEY, f)}\n\n")
        md.append("| Scope | Kind | Name | Type | # | Description |\n| --- | --- | --- | --- | --- | --- |\n")
        for scope, kind, name, typ, num, comment, line in rows:
            md.append(f"| {scope} | {kind} | [`{name}`]({common.permalink(KEY, f, line)}) | {typ} | {num} "
                      f"| {comment.replace('|', chr(92) + '|')} |\n")
        md.append("\n")
    print(common.write_gen("proto-fields.md", "".join(md)))


if __name__ == "__main__":
    main()
