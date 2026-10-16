#!/usr/bin/env python3
"""Make a citation for docs/agello/reference: a Markdown link and a ledger source record.

Untouched files link to Google's repository at the upstream base (so the link can be pasted into
an upstream issue); files our commits change link to our fork at the pin.

Usage:
  cite.py <repo> <path> <start> [<end>]          # lines
  cite.py <repo> <path> --symbol "<text>" [--context N]   # first line containing the text
  add --json for the ledger source record (repo, sha, path, lines, symbol, sha1 of the cited lines)
Repos: litert-lm, litert, ml-drift, litert-lm-shipped (see pins.json).
"""
import argparse
import hashlib
import json
import sys

import common


def source_record(key, path, start, end, symbol=None):
    text = common.show(key, path)
    lines = text.splitlines()[start - 1:end]
    url = common.permalink(key, path, start, end)
    return {
        "type": "code", "repo": key, "sha": url.split("/blob/")[1].split("/")[0], "pin": common.repo(key)["pin"],
        "path": path, "lines": [start, end], "symbol": symbol,
        "lines_sha1": hashlib.sha1("\n".join(lines).encode()).hexdigest(), "url": url,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("path")
    p.add_argument("start", nargs="?", type=int)
    p.add_argument("end", nargs="?", type=int)
    p.add_argument("--symbol")
    p.add_argument("--context", type=int, default=0, help="extra lines after the symbol line")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    if common.show(a.repo, a.path) is None:
        sys.exit(f"{a.path} not in {a.repo} at {common.repo(a.repo)['pin'][:10]}")
    if a.symbol:
        hits = common.find_symbol(a.repo, a.path, a.symbol)
        if not hits:
            sys.exit(f"'{a.symbol}' not found in {a.path}")
        if len(hits) > 1:
            print(f"note: '{a.symbol}' occurs on lines {hits}; using {hits[0]}", file=sys.stderr)
        start, end = hits[0], hits[0] + a.context
    else:
        start, end = a.start, a.end or a.start
    if a.json:
        print(json.dumps(source_record(a.repo, a.path, start, end, a.symbol)))
    else:
        print(common.md_link(a.repo, a.path, start, end))


if __name__ == "__main__":
    main()
