# agello-docs: tooling for docs/agello/reference

The reference documents LiteRT-LM, LiteRT and ML Drift at the commits in `pins.json` (our forks'
`agello-next` lines and their upstream bases), the models Agello uses, and what Agello ships. These
scripts generate its tables from sources and files, make citations, and check the result.

| Script | Does |
| --- | --- |
| `pins.json` | Repositories (local path, fork, pin, upstream, base), local model files, remote model files |
| `common.py` | Shared helpers: git, files our commits touch, permalinks |
| `cite.py` | A citation: `cite.py litert-lm runtime/core/tasks.cc --symbol "Decode("` prints a Markdown link; `--json` prints the ledger source record |
| `check_refs.py` | Checks permalinks (commit, file, line range), relative links and anchors, ledger citations (hash of the cited lines), claim tags; `--numbers` flags untagged numbers; `--repin` finds moved citations after a re-pin |
| `gen_c_api.py` | `generated/c-api-inventory.md`, from clang's AST of the public C headers |
| `gen_swift_api.sh` / `gen_swift_api.py` | `generated/swift-api-inventory.md`, from a Swift symbol graph built in a scratch folder |
| `gen_protos.py` | `generated/proto-fields.md` |
| `gen_flags.py` | `generated/cli-flags.md` |
| `gen_patch_log.py` | `generated/patch-log.md`: our commits on top of upstream, per fork |
| `gen_model_tables.py` | `generated/models/*.{md,json}`: sections, signatures, ops, composites, weights by type, per-layer attention (tools venv) |
| `gen_versions.py` | `generated/version-matrix.md` |
| `gen_binaries.py` | `generated/binaries.md`: Google's prebuilt CLiteRTLM next to ours |
| `gen_hf_configs.py` | `generated/hf/` and `generated/hf-configs.md`: official model configs at pinned revisions (`--update` moves the pins) |
| `capture_empirical.sh` / `gen_empirical.py` | `generated/kernels-by-family.md` and `generated/engine-run.md`, from engine runs per Apple GPU family |
| `regen.sh` | All of the above, then `check_refs.py` |

Citations link Google's repository at the upstream base for files our commits don't change (so
they paste into upstream issues), and our fork at the pin otherwise. Don't type line numbers by hand.

After moving a pin: update `pins.json`, run `regen.sh --all`, then `check_refs.py --repin` and
re-verify the claims it lists (see `docs/agello/reference/maintenance.md`).
