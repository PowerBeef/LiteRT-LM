# agello-bench

Measures a LiteRT-LM build on Apple silicon with fixed cases, so two builds
(arms) can be compared: Google's release binaries, a nightly, or our own
builds. It grew out of the #3913 research runner
(`research/issue-3913/runner` on the `research/3913` branch) and reproduces its
results exactly.

- `BenchKit` (library, macOS and iOS 27): the cases, their prompts and scoring,
  the machine record and the runner. Agello's harness can run the same cases
  on an iPhone.
- `agello-bench` (macOS command line): loads one engine and runs cases on it.
- `compare.py` (Python standard library): compares arms.
- `suites/`: case files. `short`, `long`, `cross4096`, `xl`, `mtp`, `speed`,
  `baseline` and `long3` are the research suites; `sweep-8k` and `sweep-32k`
  measure speed at 0.7K–4K and 0.7K–30K tokens.

## Build and run

```sh
cd tools/agello-bench
swift build -c release
swift test                      # BenchKit's unit tests (no model needed)
.build/release/agello-bench env # this machine, and the runtime binary's SHA-256

.build/release/agello-bench run --model ~/Models/agello/gemma-4-E4B-it-fp16-rope.litertlm \
  --suite suites/short.json --label A0 --cache ~/Models/agello/cache/A0-kv8192 \
  --warmup 1 --out results/A0.jsonl
```

`run` options: `--backend gpu|cpu`, `--fp32`, `--kv N` (maxNumTokens, default
8,192), `--mtp`, `--cache DIR` (`--cold` empties it first), `--repeat N`,
`--warmup N`, `--label NAME`, `--out FILE` (appends), `--hash-model`, and
`--case '<json>'` for a single case. Progress goes to stderr.

With our builds, `--cache DIR` uses `DIR/<build id>` and deletes other builds'
folders there (each holds a 2.2 GB weight copy for E4B). Two environment
variables help with memory work: `AGELLO_BENCH_HOLD=<s>` keeps the process
alive after the last case (for `footprint`, `vmmap` or `malloc_history`), and
`AGELLO_BENCH_RELEASE_TRACE=<s>` follows the footprint for that long after each
`engine-benchmark` frees its engine, printing every change of 8 MB or more.

## Arms

An arm is a LiteRT-LM checkout: its Swift wrapper and the `CLiteRTLM` binary
its `Package.swift` names. The package builds against this fork by default;
`LITERTLM_PACKAGE=/path/to/LiteRT-LM` builds it against another checkout, whose
folder must be named `LiteRT-LM` (SwiftPM names path packages after their
folder). Give each arm its own build folder:

```sh
LITERTLM_PACKAGE=~/arms/nightly/LiteRT-LM swift build -c release --scratch-path .build-nightly
```

On macOS, Google's release binaries run on WebGPU (`CLiteRTLM_mac`). The Metal
path that ships on the iPhone runs on a Mac through Agello as an iPhone app
(`PLATFORM=mac scripts/device-harness.sh` in the Agello repository) until our
own builds add a macOS Metal slice.

## Cases

A suite is a JSON array. Entries without `kind` are digit copies, the format
of the research cases:

| Kind | Fields | Measures |
| --- | --- | --- |
| `digit-copy` (default) | `id`, `sentences`, `late`, `seed` | Copy a 48-digit number placed before the filler (after it with `late`); `exact`, and the position of the first wrong digit (#3913) |
| `speed` | `id`, `sentences`, `decode`, `thinking` | Prefill the filler, then generate `decode` tokens (default 256); with `thinking` (a budget in tokens), the model thinks first, under the thinking budget's logit mask |
| `engine-benchmark` | `id`, `prefill`, `decode` | The engine's own benchmark, `LiteRTLM.benchmark` (what Agello's in-app benchmark runs): a fresh uncached engine of prefill + decode + 32 tokens, `prefill` synthetic prompt tokens (default 1,024), then `decode` tokens (default 256). A run of only these loads no chat engine |
| `helper-turn` | `id`, `sentences`, `decode`, `keep`, `helper`, `cycles` | The first reply after a helper (T3): `sentences` of history in the main conversation, a title prompt in a second one, then a short message, on the main conversation with `keep` (the runtime restores it) or on a new one seeded with the history. `helper: false` is the control without the helper; `cycles` repeats helper and message (the last is measured; the footprint after many shows a leak) |

Filler sentences average about 15.8 tokens. Every case runs in a fresh
conversation with greedy sampling, so an arm's replies should be identical
across runs, and across arms that compute the same thing.

## Records

One JSON object per line. The first keys are the research runner's (`id`,
`backend`, `activation`, `model`, `sentences`, `late`, `seed`, `prefill`,
`decode`, `exact`, `expected`, `output`, `seconds`, `mtp`, `prefillTPS`,
`decodeTPS`, `firstMismatch`, `firstMismatchPosition`). agello-bench adds:

| Key | Meaning |
| --- | --- |
| `schema`, `kind`, `label`, `run`, `rep`, `warmup` | Record format, case kind, arm, process run, repetition, warm-up flag |
| `kv` | KV cache capacity (maxNumTokens) |
| `ttft` | The engine's time to first token |
| `thermal` | `ProcessInfo.thermalState` after the case |
| `footprintMB`, `peakFootprintMB` | The process's physical footprint after the case, and its peak so far |
| `loadSeconds`, `cache` | Engine load time, and whether the cache folder was `cold`, `warm` or `off` |
| `modelSHA256` | With `--hash-model` |
| `machine` | Hardware, chip, OS, memory, `isiOSAppOnMac`, low power mode, GPU name, Apple GPU family and Metal working set, and the runtime binary's path and SHA-256 |

## Comparing arms

```sh
python3 compare.py results/A0.jsonl results/A2.jsonl --base A0
```

It prints each arm's setup, digit-copy fidelity and identical replies against
the base arm, then, for each metric (`--metrics`, default `prefillTPS
decodeTPS ttft`), the median and IQR per case, each arm's change against the
base with a 95% bootstrap interval, the geometric mean change, and the noise
band: the base arm's even runs against its odd ones.

## Protocol

- Run arms in ABAB order, one process per arm and round, so drift and heat
  hit both arms alike, with `--warmup 1` (the first case of a fresh process
  runs slow).
- Speed: five rounds per arm. Greedy fidelity: one round is enough.
- Plug the Mac in, close other GPU work, and run under `caffeinate -dims`.
  Never measure during a Bazel build or an Agello harness run.
- A change inside the noise band isn't a change. Repeat runs of one arm stay
  within 3%.
