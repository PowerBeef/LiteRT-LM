#!/usr/bin/env python3
"""generated/patch-log.md: every commit of ours between the upstream base and the pin, per fork, with files."""
import common

REPOS = ["litert-lm", "litert", "ml-drift", "litert-lm-shipped"]


def main():
    md = [common.gen_header("gen_patch_log.py", REPOS), "# Our commits on top of upstream\n\n"]
    for key in REPOS:
        r = common.repo(key)
        log = common.git(key, "log", "--reverse", "--format=%H%x09%ad%x09%s", "--date=short",
                         f"{r['base']}..{r['pin']}").strip().splitlines()
        md.append(f"## {key}: `{r['fork']}` `{r['branch']}`\n\n{len(log)} commits from {r['upstream']} "
                  f"`{r['base'][:10]}` ({r['base_label']}) to `{r['pin'][:10]}`; "
                  f"{len(common.touched(key))} files differ.\n\n| Commit | Date | Subject | Files |\n| --- | --- | --- | --- |\n")
        for row in log:
            sha, date, subject = row.split("\t", 2)
            files = common.git(key, "show", "--name-only", "--format=", sha).split()
            shown = ", ".join(f"`{f}`" for f in files[:6]) + (f" (+{len(files) - 6})" if len(files) > 6 else "")
            md.append(f"| [`{sha[:10]}`](https://github.com/{r['fork']}/commit/{sha}) | {date} "
                      f"| {subject.replace('|', chr(92) + '|')} | {shown} |\n")
        md.append("\n<details><summary>All files that differ from upstream</summary>\n\n"
                  + "\n".join(f"- `{f}`" for f in sorted(common.touched(key))) + "\n\n</details>\n\n")
    print(common.write_gen("patch-log.md", "".join(md)))


if __name__ == "__main__":
    main()
