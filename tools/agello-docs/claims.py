#!/usr/bin/env python3
"""Claims and verdicts for docs/agello/reference (the research runs; see _work/style.md).

A research group (docs/agello/reference/_work/groups.json) writes its claims to
_work/claims/<group>.jsonl; verifiers write _work/verdicts/<group>.<batch>.jsonl; merge writes
ledger/<group>.jsonl, which check_refs.py reads.

  claims.py check <group>                    validate the claims file and the claim tags in the group's pages
  claims.py list <group>                     JSON list of {id, tier, kind}
  claims.py show <group> --ids A,B,...       the claims (text and sources only) with the cited code lines
  claims.py check-verdicts <group> <batch> --ids A,B,...   validate a verdict file
  claims.py merge <group>                    claims + verdicts -> ledger/<group>.jsonl (verdict files are read in
                                             name order and later ones win: audits use batch "z-audit")
  claims.py check-applied <group>            after merge and edits: verified/corrected tier-A claims are tagged,
                                             wrong/unverifiable ones are not
  claims.py stats [<group> ...]              verdict counts by tier
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

import common
import check_refs

WORK = os.path.join(common.REF_DIR, "_work")
CLAIMS_DIR = os.path.join(WORK, "claims")
VERDICTS_DIR = os.path.join(WORK, "verdicts")
REPO_ROOT = os.path.normpath(os.path.join(common.REF_DIR, "..", "..", ".."))
AGELLO = "/Users/patricedery/Coding projects/iRT"
GROUPS = json.load(open(os.path.join(WORK, "groups.json")))["groups"]

KINDS = {"code-structure", "code-behavior", "apple-path", "number", "model-structure", "measurement",
         "external", "process"}
VERDICTS = {"verified", "corrected", "wrong", "unverifiable"}
ID_RE = re.compile(r"^([A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]*)*)-(\d{3})$")
TAG = re.compile(r"<!--\s*c:([A-Z0-9-]+)\s*-->")


def group(name):
    if name not in GROUPS:
        sys.exit(f"unknown group {name}; groups: {', '.join(GROUPS)}")
    return GROUPS[name]


def read_jsonl(path, errors):
    out = []
    if not os.path.exists(path):
        errors.append(f"missing {os.path.relpath(path, common.REF_DIR)}")
        return out
    for n, line in enumerate(open(path), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            errors.append(f"{os.path.basename(path)}:{n}: not JSON ({e})")
    return out


def load_claims(name, errors=None):
    errors = [] if errors is None else errors
    return read_jsonl(os.path.join(CLAIMS_DIR, f"{name}.jsonl"), errors)


def check_source(s, where, errors):
    t = s.get("type")
    if t == "code":
        key = s.get("repo")
        if key not in common.PINS["repos"]:
            errors.append(f"{where}: unknown repo {key!r}")
            return
        sha = s.get("sha") or common.repo(key)["pin"]
        r = common.repo(key)
        if sha not in (r["pin"], r["base"]):
            errors.append(f"{where}: sha {sha[:10]} is not the pin or base of {key} (make it with cite.py --json)")
        text = common.show(key, s.get("path", ""), sha)
        if text is None:
            errors.append(f"{where}: {s.get('path')} not in {key}@{sha[:10]}")
            return
        lines = s.get("lines") or []
        if len(lines) != 2 or not all(isinstance(x, int) for x in lines) or lines[0] < 1 or lines[1] < lines[0]:
            errors.append(f"{where}: bad lines {lines}")
            return
        total = len(text.splitlines())
        if lines[1] > total:
            errors.append(f"{where}: lines {lines} past the end of {s['path']} ({total})")
            return
        cited = "\n".join(text.splitlines()[lines[0] - 1:lines[1]])
        if s.get("lines_sha1") != hashlib.sha1(cited.encode()).hexdigest():
            errors.append(f"{where}: lines_sha1 doesn't match {s['path']}:{lines[0]}-{lines[1]} (make it with cite.py --json)")
        if s.get("symbol") and s["symbol"] not in cited:
            errors.append(f"{where}: symbol {s['symbol']!r} not in {s['path']}:{lines[0]}-{lines[1]}")
        if not s.get("url"):
            errors.append(f"{where}: no url")
    elif t == "web":
        for k in ("url", "publisher", "version", "retrieved", "quote"):
            if not s.get(k):
                errors.append(f"{where}: web source without {k}")
        if len(s.get("quote", "").split()) > 25:
            errors.append(f"{where}: quote longer than 25 words")
    elif t == "empirical":
        p = s.get("path", "")
        if not os.path.exists(os.path.join(common.REF_DIR, p)):
            errors.append(f"{where}: empirical path {p!r} not under docs/agello/reference")
        if not s.get("detail"):
            errors.append(f"{where}: empirical source without detail")
    elif t == "results":
        p = s.get("path", "")
        if not os.path.exists(os.path.join(REPO_ROOT, p)):
            errors.append(f"{where}: results path {p!r} not in the fork checkout")
        if not s.get("conditions"):
            errors.append(f"{where}: results source without conditions")
    elif t == "agello":
        commit = s.get("commit", "")
        r = subprocess.run(["git", "-C", AGELLO, "show", f"{commit}:{s.get('path', '')}"], capture_output=True)
        if r.returncode != 0:
            errors.append(f"{where}: {s.get('path')} not in Agello@{commit[:10]}")
        elif s.get("lines"):
            n = len(r.stdout.decode("utf-8", "replace").splitlines())
            if s["lines"][1] > n:
                errors.append(f"{where}: lines past the end of Agello {s['path']}")
    else:
        errors.append(f"{where}: unknown source type {t!r}")


def page_tags(name):
    tags = {}
    for p in group(name)["pages"]:
        path = os.path.join(common.REF_DIR, p)
        if not os.path.exists(path):
            continue
        for n, line in enumerate(open(path), 1):
            for cid in TAG.findall(line):
                tags.setdefault(cid, []).append(f"{p}:{n}")
    return tags


def ledger_verdicts():
    """Claim ID -> verdict, over every ledger file (pages may tag other groups' verified claims)."""
    out = {}
    for f in glob.glob(os.path.join(common.LEDGER_DIR, "*.jsonl")):
        for line in open(f):
            if line.strip():
                c = json.loads(line)
                out[c["id"]] = c.get("verdict")
    return out


def all_ids(exclude):
    ids = {}
    for f in glob.glob(os.path.join(CLAIMS_DIR, "*.jsonl")):
        g = os.path.basename(f)[:-6]
        if g == exclude:
            continue
        for c in load_claims(g):
            ids[c.get("id")] = g
    return ids


def cmd_check(name):
    g = group(name)
    errors = []
    claims = load_claims(name, errors)
    others = all_ids(name)
    seen = set()
    for i, c in enumerate(claims, 1):
        cid = c.get("id", f"line {i}")
        where = f"{name}.jsonl {cid}"
        m = ID_RE.match(c.get("id", ""))
        if not m:
            errors.append(f"{where}: id must look like LM-ENG-001")
        elif m.group(1) not in g["prefixes"]:
            errors.append(f"{where}: prefix {m.group(1)} not one of {g['prefixes']}")
        if cid in seen:
            errors.append(f"{where}: duplicate id")
        if cid in others:
            errors.append(f"{where}: id also used by group {others[cid]}")
        seen.add(cid)
        if c.get("page") not in g["pages"]:
            errors.append(f"{where}: page {c.get('page')!r} not one of the group's pages")
        else:
            path = os.path.join(common.REF_DIR, c["page"])
            if not os.path.exists(path):
                errors.append(f"{where}: page {c['page']} doesn't exist yet")
            elif c.get("anchor") and c["anchor"] not in check_refs.anchors(path):
                errors.append(f"{where}: no heading #{c['anchor']} in {c['page']}")
        if not c.get("text"):
            errors.append(f"{where}: no text")
        if c.get("tier") not in ("A", "B"):
            errors.append(f"{where}: tier must be A or B")
        if c.get("kind") not in KINDS:
            errors.append(f"{where}: kind must be one of {sorted(KINDS)}")
        if c.get("kind") == "number" and not c.get("formula"):
            errors.append(f"{where}: a number needs a formula")
        ap = c.get("applies_to") or {}
        if ap.get("platform") not in ("apple", "all") or ap.get("build") not in ("documented", "shipped", "both"):
            errors.append(f"{where}: applies_to needs platform apple|all and build documented|shipped|both")
        srcs = c.get("sources") or []
        if not srcs:
            errors.append(f"{where}: no sources")
        for j, s in enumerate(srcs):
            check_source(s, f"{where} source {j + 1}", errors)
    tags = page_tags(name)
    by_id = {c.get("id"): c for c in claims}
    elsewhere = ledger_verdicts()
    for cid, wheres in tags.items():
        c = by_id.get(cid)
        if c is None:
            if elsewhere.get(cid) in ("verified", "corrected") or (cid in others and cid not in elsewhere):
                continue  # another group's claim: verified in the ledger, or not merged yet
            errors.append(f"tag {cid} at {wheres[0]} is neither a claim of {name} nor a verified ledger claim")
        elif c.get("tier") != "A":
            errors.append(f"tag {cid} at {wheres[0]}: only tier-A claims carry tags")
    for c in claims:
        if c.get("tier") == "A" and c.get("id") not in tags:
            errors.append(f"tier-A claim {c.get('id')} has no <!-- c:{c.get('id')} --> tag in {c.get('page')}")
    for e in errors:
        print("error", e)
    a = sum(c.get("tier") == "A" for c in claims)
    print(f"{name}: {len(claims)} claims ({a} tier A, {len(claims) - a} tier B), {len(errors)} errors")
    return 1 if errors else 0


def cmd_list(name):
    print(json.dumps([{"id": c["id"], "tier": c["tier"], "kind": c["kind"]} for c in load_claims(name)]))


def cmd_show(name, ids):
    by_id = {c["id"]: c for c in load_claims(name)}
    missing = [i for i in ids if i not in by_id]
    if missing:
        sys.exit(f"not in {name}.jsonl: {', '.join(missing)}")
    for cid in ids:
        c = by_id[cid]
        shown = {k: c[k] for k in ("id", "tier", "kind", "text", "applies_to", "formula", "conflicts_with", "sources")
                 if k in c}
        print("=" * 100)
        print(json.dumps(shown, indent=1, ensure_ascii=False))
        for s in c.get("sources", []):
            if s.get("type") != "code":
                continue
            text = common.show(s["repo"], s["path"], s.get("sha")) or ""
            a, b = s["lines"]
            print(f"--- {s['repo']}@{(s.get('sha') or '')[:10]} {s['path']}:{a}-{b}")
            lines = text.splitlines()
            for n in range(a, min(b, a + 59) + 1):
                if n <= len(lines):
                    print(f"{n:6} {lines[n - 1]}")
            if b - a > 59:
                print(f"       ... ({b - a - 59} more lines)")


def verdict_file(name, batch):
    return os.path.join(VERDICTS_DIR, f"{name}.{batch}.jsonl")


def cmd_check_verdicts(name, batch, ids):
    errors = []
    rows = read_jsonl(verdict_file(name, batch), errors)
    got = {}
    for r in rows:
        cid = r.get("id")
        where = f"{name}.{batch}.jsonl {cid}"
        if r.get("verdict") not in VERDICTS:
            errors.append(f"{where}: verdict must be one of {sorted(VERDICTS)}")
        if not r.get("note"):
            errors.append(f"{where}: no note (say what you checked)")
        if r.get("verdict") == "corrected" and not r.get("corrected_text"):
            errors.append(f"{where}: corrected without corrected_text")
        for j, s in enumerate(r.get("corrected_sources") or []):
            check_source(s, f"{where} corrected source {j + 1}", errors)
        if cid in got:
            errors.append(f"{where}: duplicate")
        got[cid] = r
    if ids:
        missing = [i for i in ids if i not in got]
        extra = [i for i in got if i not in ids]
        if missing:
            errors.append(f"no verdict for: {', '.join(missing)}")
        if extra:
            errors.append(f"verdicts for claims not in the batch: {', '.join(extra)}")
    for e in errors:
        print("error", e)
    counts = {}
    for r in got.values():
        counts[r.get("verdict")] = counts.get(r.get("verdict"), 0) + 1
    print(f"{name}.{batch}: {len(got)} verdicts {counts}, {len(errors)} errors")
    return 1 if errors else 0


def merged(name):
    verdicts = {}
    for f in sorted(glob.glob(os.path.join(VERDICTS_DIR, f"{name}.*.jsonl"))):
        batch = os.path.basename(f)[len(name) + 1:-6]
        for r in read_jsonl(f, []):
            verdicts[r.get("id")] = dict(r, batch=batch)
    out = []
    for c in load_claims(name):
        c = dict(c)
        v = verdicts.get(c["id"])
        if v is None:
            c["verdict"] = "unsampled" if c.get("tier") == "B" else "unverified"
        else:
            c["verdict"] = v["verdict"]
            c["verified_in"] = f"{name}.{v['batch']}"
            c["verifier_note"] = v.get("note", "")
            if v["verdict"] == "corrected":
                c["original_text"] = c["text"]
                c["text"] = v["corrected_text"]
                if v.get("corrected_sources"):
                    c["original_sources"] = c["sources"]
                    c["sources"] = v["corrected_sources"]
        out.append(c)
    return out


def cmd_merge(name):
    rows = merged(name)
    os.makedirs(common.LEDGER_DIR, exist_ok=True)
    path = os.path.join(common.LEDGER_DIR, f"{name}.jsonl")
    with open(path, "w") as f:
        for c in rows:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {os.path.relpath(path, common.REF_DIR)}: {len(rows)} claims")
    cmd_stats([name])


def cmd_check_applied(name):
    path = os.path.join(common.LEDGER_DIR, f"{name}.jsonl")
    if not os.path.exists(path):
        sys.exit(f"run claims.py merge {name} first")
    rows = [json.loads(x) for x in open(path) if x.strip()]
    tags = page_tags(name)
    errors = []
    for c in rows:
        tagged = c["id"] in tags
        if c.get("tier") == "A" and c["verdict"] in ("verified", "corrected") and not tagged:
            errors.append(f"{c['id']} ({c['verdict']}) lost its tag in {c.get('page')}")
        if c["verdict"] in ("wrong", "unverifiable", "unsampled") and tagged:
            errors.append(f"{c['id']} is {c['verdict']} but still tagged at {tags[c['id']][0]}")
    known = {c["id"] for c in rows}
    elsewhere = ledger_verdicts()
    for cid, wheres in tags.items():
        if cid not in known and elsewhere.get(cid) not in ("verified", "corrected"):
            errors.append(f"tag {cid} at {wheres[0]} is neither in ledger/{name}.jsonl nor a verified ledger claim")
    for e in errors:
        print("error", e)
    print(f"{name}: {len(rows)} ledger claims, {len(tags)} tags, {len(errors)} errors")
    return 1 if errors else 0


def cmd_stats(names):
    names = names or sorted(os.path.basename(f)[:-6] for f in glob.glob(os.path.join(CLAIMS_DIR, "*.jsonl")))
    cols = ["verified", "corrected", "wrong", "unverifiable", "unsampled", "unverified"]
    print(f"{'group':18} {'tier':4} " + " ".join(f"{c:>12}" for c in cols))
    for name in names:
        rows = merged(name)
        for tier in ("A", "B"):
            cnt = {k: 0 for k in cols}
            for c in rows:
                if c.get("tier") == tier:
                    cnt[c["verdict"]] += 1
            print(f"{name:18} {tier:4} " + " ".join(f"{cnt[c]:>12}" for c in cols))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd in ("check", "list", "merge", "check-applied"):
        sub.add_parser(cmd).add_argument("group")
    s = sub.add_parser("show")
    s.add_argument("group")
    s.add_argument("--ids", required=True)
    v = sub.add_parser("check-verdicts")
    v.add_argument("group")
    v.add_argument("batch")
    v.add_argument("--ids", default="")
    st = sub.add_parser("stats")
    st.add_argument("groups", nargs="*")
    a = p.parse_args()
    ids = [x for x in getattr(a, "ids", "").replace(" ", "").split(",") if x] if hasattr(a, "ids") else []
    if a.cmd == "check":
        sys.exit(cmd_check(a.group))
    if a.cmd == "list":
        cmd_list(a.group)
    elif a.cmd == "show":
        cmd_show(a.group, ids)
    elif a.cmd == "check-verdicts":
        sys.exit(cmd_check_verdicts(a.group, a.batch, ids))
    elif a.cmd == "merge":
        cmd_merge(a.group)
    elif a.cmd == "check-applied":
        sys.exit(cmd_check_applied(a.group))
    elif a.cmd == "stats":
        cmd_stats(a.groups)


if __name__ == "__main__":
    main()
