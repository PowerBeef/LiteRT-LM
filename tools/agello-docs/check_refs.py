#!/usr/bin/env python3
"""Check docs/agello/reference: links, anchors, cited code ranges, ledger claims, claim tags.

  check_refs.py            all checks; exit 1 on errors
  check_refs.py --numbers  also warn about numbers with units that carry no fact ID or claim tag
  check_refs.py --repin    after moving the pins: find each ledger citation's symbol at the new pin,
                           report claims whose cited code changed (they need re-verification)
  check_refs.py --only litert-lm/c-api.md ...   check only these pages (paths relative to reference/)
  check_refs.py --no-tags  skip the claim-tag check (pages whose claims aren't in the ledger yet)

Checks:
  - GitHub blob permalinks: the commit is a pin or an upstream base in pins.json, the file exists
    there, and the line range is inside the file.
  - Relative links: the target exists; #anchors match a heading (GitHub's slug rules).
  - Ledger (ledger/*.jsonl): each code source's lines still hash to lines_sha1 at its commit, and its
    symbol (if any) is in those lines.
  - Claim tags (<!-- c:ID -->) in pages refer to ledger claims whose verdict is verified or corrected.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

import common

LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")
BLOB = re.compile(r"https://github\.com/([^/]+/[^/]+)/blob/([0-9a-f]{40})/([^#]+)(?:#L(\d+)(?:-L(\d+))?)?$")
TAG = re.compile(r"<!--\s*c:([A-Z0-9-]+)\s*-->")
NUMBER = re.compile(r"\b\d[\d,.]*\s?(?:GB|MB|KB|GiB|MiB|tok/s|ms|µs|%|tokens|K tokens)\b")


def slug(heading):
    s = heading.strip().lower()
    s = re.sub(r"[`*_]", "", s)
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def anchors(path):
    out, seen = set(), {}
    in_code = False
    for line in open(path):
        if line.startswith("```"):
            in_code = not in_code
        if in_code or not line.startswith("#"):
            continue
        base = slug(line.lstrip("#"))
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.add(base if n == 0 else f"{base}-{n}")
    for m in re.finditer(r'<a (?:name|id)="([^"]+)"', open(path).read()):
        out.add(m.group(1))
    return out


def commit_key(slug_, sha):
    for key, r in common.PINS["repos"].items():
        if (slug_ == r["fork"] and sha == r["pin"]) or (slug_ == r["upstream"] and sha == r["base"]):
            return key
    return None


ONLY = None


def pages():
    found = sorted(glob.glob(os.path.join(common.REF_DIR, "**", "*.md"), recursive=True))
    if ONLY:
        found = [p for p in found if os.path.relpath(p, common.REF_DIR) in ONLY]
    return found


def ledger():
    claims = {}
    for f in sorted(glob.glob(os.path.join(common.LEDGER_DIR, "*.jsonl"))):
        for n, line in enumerate(open(f), 1):
            if line.strip():
                c = json.loads(line)
                claims[c["id"]] = (c, f"{os.path.basename(f)}:{n}")
    return claims


def check_links(errors):
    for page in pages():
        rel = os.path.relpath(page, common.REF_DIR)
        in_code = False
        for n, line in enumerate(open(page), 1):
            if line.startswith("```"):
                in_code = not in_code
            if in_code:
                continue
            for text, url in LINK.findall(line):
                where = f"{rel}:{n}"
                m = BLOB.match(url)
                if m:
                    slug_, sha, path, start, end = m.groups()
                    key = commit_key(slug_, sha)
                    if key is None:
                        errors.append(f"{where}: permalink to {slug_}@{sha[:10]} is not a pin or base")
                        continue
                    content = common.show(key, path, sha)
                    if content is None:
                        errors.append(f"{where}: {path} not at {slug_}@{sha[:10]}")
                        continue
                    count = len(content.splitlines())
                    for x in (start, end):
                        if x and int(x) > count:
                            errors.append(f"{where}: line {x} past the end of {path} ({count} lines)")
                elif url.startswith("http"):
                    continue
                elif url.startswith("#"):
                    if url[1:] not in anchors(page):
                        errors.append(f"{where}: no anchor {url}")
                else:
                    target, _, anchor = url.partition("#")
                    tpath = os.path.normpath(os.path.join(os.path.dirname(page), target))
                    if not os.path.exists(tpath):
                        errors.append(f"{where}: missing {target}")
                    elif anchor and tpath.endswith(".md") and anchor not in anchors(tpath):
                        errors.append(f"{where}: no anchor #{anchor} in {target}")


def check_ledger(errors, claims):
    for cid, (c, where) in claims.items():
        for s in c.get("sources", []):
            if s.get("type") != "code":
                continue
            key = s.get("repo")
            if key not in common.PINS["repos"]:
                errors.append(f"{where} {cid}: unknown repo {key}")
                continue
            sha = s.get("sha") or common.repo(key)["pin"]
            text = common.show(key, s["path"], sha)
            if text is None:
                errors.append(f"{where} {cid}: {s['path']} not at {key}@{sha[:10]}")
                continue
            a, b = s["lines"]
            cited = "\n".join(text.splitlines()[a - 1:b])
            if s.get("lines_sha1") and hashlib.sha1(cited.encode()).hexdigest() != s["lines_sha1"]:
                errors.append(f"{where} {cid}: cited lines {s['path']}:{a}-{b} changed")
            if s.get("symbol") and s["symbol"] not in cited:
                errors.append(f"{where} {cid}: '{s['symbol']}' not in {s['path']}:{a}-{b}")


def check_tags(errors, claims, warn_numbers):
    warnings = 0
    for page in pages():
        rel = os.path.relpath(page, common.REF_DIR)
        if rel.startswith(("generated", "_work")):
            continue
        in_code = False
        for n, line in enumerate(open(page), 1):
            if line.startswith("```"):
                in_code = not in_code
            for cid in TAG.findall(line):
                c = claims.get(cid)
                if c is None:
                    errors.append(f"{rel}:{n}: claim {cid} not in the ledger")
                elif c[0].get("verdict") not in ("verified", "corrected"):
                    errors.append(f"{rel}:{n}: claim {cid} is {c[0].get('verdict', 'unverified')}")
            if warn_numbers and not in_code and NUMBER.search(line) and not TAG.search(line) \
                    and not re.search(r"\bF-\d+", line) and not line.lstrip().startswith(("|", ">")):
                warnings += 1
                print(f"warning {rel}:{n}: number without a fact ID or claim tag: {line.strip()[:100]}")
    return warnings


def repin(claims):
    """For code sources whose commit is no longer a pin or base, find the symbol at the current pin."""
    changed = 0
    for cid, (c, where) in claims.items():
        for s in c.get("sources", []):
            if s.get("type") != "code" or s.get("repo") not in common.PINS["repos"]:
                continue
            r = common.repo(s["repo"])
            if s.get("sha") in (r["pin"], r["base"]):
                continue
            hits = common.find_symbol(s["repo"], s["path"], s["symbol"]) if s.get("symbol") else []
            text = common.show(s["repo"], s["path"]) or ""
            status = "file gone" if not text else ("symbol gone" if s.get("symbol") and not hits else "moved")
            if hits:
                a = hits[0]
                b = a + (s["lines"][1] - s["lines"][0])
                same = hashlib.sha1("\n".join(text.splitlines()[a - 1:b]).encode()).hexdigest() == s.get("lines_sha1")
                status = "unchanged" if same else "changed: re-verify"
                print(f"{cid}: {s['path']}:{s['lines'][0]} -> {a}  {status}")
            else:
                print(f"{cid}: {s['path']}:{s['lines'][0]}  {status}: re-verify")
            changed += status != "unchanged"
    print(f"{changed} citations need re-verification")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--numbers", action="store_true")
    p.add_argument("--repin", action="store_true")
    p.add_argument("--only", nargs="+", help="pages to check, relative to docs/agello/reference")
    p.add_argument("--no-tags", action="store_true")
    a = p.parse_args()
    global ONLY
    if a.only:
        ONLY = {os.path.normpath(x) for x in a.only}
        missing = [x for x in ONLY if not os.path.exists(os.path.join(common.REF_DIR, x))]
        if missing:
            sys.exit(f"no such page: {', '.join(sorted(missing))}")
    claims = ledger()
    if a.repin:
        repin(claims)
        return
    errors = []
    check_links(errors)
    if not a.only:
        check_ledger(errors, claims)
    warnings = 0 if a.no_tags else check_tags(errors, claims, a.numbers)
    for e in errors:
        print("error", e)
    print(f"{len(pages())} pages, {len(claims)} ledger claims: {len(errors)} errors, {warnings} number warnings")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
