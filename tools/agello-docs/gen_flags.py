#!/usr/bin/env python3
"""generated/cli-flags.md: the ABSL flags of LiteRT-LM's command-line tools at the pin."""
import re

import common

KEY = "litert-lm"
FILES = ["runtime/engine/shared_flags.cc", "runtime/engine/litert_lm_main.cc",
         "runtime/engine/litert_lm_advanced_main.cc", "runtime/engine/embedding_litert_lm_main.cc"]
# ABSL_FLAG(type, name, default, "help" "help...");  spans lines
FLAG = re.compile(r"ABSL_FLAG\(\s*(.+?)\s*,\s*(\w+)\s*,\s*(.*?)\s*,\s*((?:\"(?:[^\"\\]|\\.)*\"\s*)+)\)\s*;", re.S)


def main():
    md = [common.gen_header("gen_flags.py", [KEY]), "# LiteRT-LM command-line flags\n\n",
          "`litert_lm_main` and `litert_lm_advanced_main` share `shared_flags.cc`; each binary may add its own.\n\n"]
    total = 0
    for f in FILES:
        text = common.show(KEY, f)
        if text is None:
            continue
        flags = []
        for m in FLAG.finditer(text):
            typ, name, default, helptext = m.groups()
            line = text[:m.start()].count("\n") + 1
            helptext = " ".join(re.findall(r'"((?:[^"\\]|\\.)*)"', helptext))
            flags.append((name, typ, " ".join(default.split()), helptext, line))
        if not flags:
            continue
        total += len(flags)
        md.append(f"## `{f}` ({len(flags)} flags)\n\n| Flag | Type | Default | Help |\n| --- | --- | --- | --- |\n")
        for name, typ, default, helptext, line in flags:
            md.append(f"| [`--{name}`]({common.permalink(KEY, f, line)}) | `{typ}` | `{default}` "
                      f"| {helptext.replace('|', chr(92) + '|')} |\n")
        md.append("\n")
    md.append(f"Total: {total} flags.\n")
    print(common.write_gen("cli-flags.md", "".join(md)))


if __name__ == "__main__":
    main()
