#!/usr/bin/env python3
"""generated/hf/<repo>/... and generated/hf-configs.md: the official model configurations, pinned.

  gen_hf_configs.py            fetch the small config files of the repos in pins.json "hf_repos" at
                               their pinned revision, and tabulate every scalar config key across models
  gen_hf_configs.py --update   first move each pin to the repo's current revision (writes pins.json)

Only small JSON and template files are fetched (no weights, no tokenizer.json). They're Google's
files, kept verbatim with their source revision; see each repo's license.
"""
import json
import os
import sys
import urllib.request

import common

FILES = ["config.json", "generation_config.json", "processor_config.json", "preprocessor_config.json",
         "tokenizer_config.json", "chat_template.jinja", "modules.json", "config_sentence_transformers.json",
         "sentence_bert_config.json", "1_Pooling/config.json", "2_Normalize/config.json"]
MAX_BYTES = 2 << 20


def api(repo):
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=60) as r:
        return json.load(r)


def fetch(repo, sha, name):
    url = f"https://huggingface.co/{repo}/resolve/{sha}/{name}"
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read(MAX_BYTES + 1)
    return None if len(data) > MAX_BYTES else data


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list) and len(v) > 12:
            out[key] = f"list of {len(v)}: " + ", ".join(f"{x}×{v.count(x)}" for x in sorted(set(map(str, v)))) \
                if all(not isinstance(x, (dict, list)) for x in v) else f"list of {len(v)}"
        else:
            out[key] = v
    return out


def main():
    pins = common.PINS
    repos = pins["hf_repos"]
    if "--update" in sys.argv:
        for repo in repos:
            repos[repo] = api(repo)["sha"]
        json.dump(pins, open(os.path.join(common.HERE, "pins.json"), "w"), indent=2)
        open(os.path.join(common.HERE, "pins.json"), "a").write("\n")
    out_dir = os.path.join(common.GEN_DIR, "hf")
    configs = {}
    for repo, sha in repos.items():
        siblings = {s["rfilename"] for s in api(repo).get("siblings", [])}
        folder = os.path.join(out_dir, repo.replace("/", "__"))
        os.makedirs(folder, exist_ok=True)
        for name in FILES:
            if name not in siblings:
                continue
            data = fetch(repo, sha, name)
            if data is None:
                continue
            path = os.path.join(folder, name.replace("/", "__"))
            open(path, "wb").write(data)
            if name == "config.json":
                configs[repo] = json.loads(data)
        open(os.path.join(folder, "SOURCE.txt"), "w").write(f"https://huggingface.co/{repo}/tree/{sha}\n")
        print(f"{repo} @ {sha[:12]}")

    flat = {r: flatten(c) for r, c in configs.items()}
    keys = sorted({k for f in flat.values() for k in f})
    md = [common.gen_header("gen_hf_configs.py", ["Hugging Face model repositories"]),
          "# Official model configurations\n\n",
          "Every scalar key of each `config.json`, at the pinned revision (files in `hf/`). Long lists are "
          "summarized as value×count.\n\n| Repository | Revision |\n| --- | --- |\n"]
    md += [f"| [{r}](https://huggingface.co/{r}/tree/{s}) | `{s[:12]}` |\n" for r, s in repos.items()]
    groups = {}
    for k in keys:
        groups.setdefault(k.split(".")[0] if "." in k else "(top level)", []).append(k)
    names = list(flat)
    short = [n.split("/")[1] for n in names]
    for g, ks in groups.items():
        md.append(f"\n## {g}\n\n| Key | " + " | ".join(short) + " |\n| --- |" + " --- |" * len(short) + "\n")
        for k in ks:
            vals = [str(flat[n].get(k, "")).replace("|", "\\|") for n in names]
            md.append(f"| `{k}` | " + " | ".join(vals) + " |\n")
    print(common.write_gen("hf-configs.md", "".join(md)))


if __name__ == "__main__":
    main()
