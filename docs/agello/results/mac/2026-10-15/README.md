# Mac results, 2026-10-15

Raw records behind tracks/T4-attention.md (step 2, the sliding window), T3-helpers.md and
T2-host-costs.md. Same machine and model as `../2026-10-14/README.md`; home-directory paths
replaced by `~`.

## `bench/`

| File | What |
| --- | --- |
| `t4b-fid.jsonl` | Digit-copy suites at a 32K window with the decode window (LiteRT `31386317`, ML Drift `6ae7c68`), label `win`; compare with `../2026-10-14/bench/t4-fid.jsonl` (`on`) |
| `t4c-fid.jsonl` | Same, with the prefill window too (ML Drift `2184a7f`), label `prefillwin` |
| `t4b-ab-*.jsonl` | The window off and on in one binary (`AGELLO_BOUNDED_WINDOW=0` for `off`), ABAB, 1 warm-up and 3 repetitions per arm: 658 tokens at 8K (MTP off and on), 2K at 32K, 16K at 32K |
| `t3-helper.jsonl` | `helper-turn` at a 16K window, MTP on: reseed against keep at 2K, 4K and 8K of history, build `62b72973cf80` (lock-path snapshots) |
| `t3-raw.jsonl` | `helper-turn-check` with raw snapshots (P3.3), MTP on and off |
| `t3-check-{pre,post,post2}.jsonl` | `helper-turn-check`, 16K, MTP on: P3.3 (`3c34537772ce`), + no orphan snapshot, + mapped snapshots |
| `t3-cycles-{pre,post,post2}.jsonl` | `helper-turn-cycles` (1, 25, 100 rounds on one conversation), same three builds |
| `t11-{shared,perkv}.jsonl` | `short` digit copies at 8K, 4K, 16K and 8K in one cache folder, with the content key and with the upstream key (`AGELLO_SHARED_WEIGHT_CACHE=0`) |
| `p6-{,mtp-}{p6pre,p6post}.jsonl` | `thinking-mask` at 8K, MTP off and on, 1 warm-up and 5 repetitions, before and after the FP16 greedy path |
| `t6-{plain,mtp}.jsonl` | One 700-token speed case at telemetry level 2 (spans, GPU time per run), MTP off and on |
| `t2-think{,-mtp}-{pre,post,post2}.jsonl` | `thinking-mask` at 8K, MTP off and on, 1 warm-up and 3 repetitions: before T2, + mask skip and per-step changes, + `GetVocabSize` |

## `agello/`

| File | Build | Suite |
| --- | --- | --- |
| `inference-t4b.json`, `probes-t4b.json` | the decode window | 21/21; 12/12 (3 lab-model probes skipped) |
| `inference-t4c.json` | the prefill window too | 21/21 |
| `inference-p4.json`, `probes-p4.json` | `c2a5c0a3df2a` (T3 and T2 changes) | 21/21; 11/11 |
| `probes-t11.json` | `d1c570842441` (one weight cache for every size) | 11/11, probe 14 included |
| `calibrate-final.json`, `inference-final.json`, `probes-final.json` | `c5460a024e71` (everything above, FP16 greedy path) | calibration at 4K–32K; 21/21; 11/11 |
