# Mac results, 2026-10-15

Raw records behind tracks/T4-attention.md (step 2, the sliding window). Same machine and model as
`../2026-10-14/README.md`; home-directory paths replaced by `~`.

## `bench/`

| File | What |
| --- | --- |
| `t4b-fid.jsonl` | Digit-copy suites at a 32K window with the decode window (LiteRT `31386317`, ML Drift `6ae7c68`), label `win`; compare with `../2026-10-14/bench/t4-fid.jsonl` (`on`) |
| `t4c-fid.jsonl` | Same, with the prefill window too (ML Drift `2184a7f`), label `prefillwin` |
| `t4b-ab-*.jsonl` | The window off and on in one binary (`AGELLO_BOUNDED_WINDOW=0` for `off`), ABAB, 1 warm-up and 3 repetitions per arm: 658 tokens at 8K (MTP off and on), 2K at 32K, 16K at 32K |

## `agello/`

| File | Build | Suite |
| --- | --- | --- |
| `inference-t4b.json`, `probes-t4b.json` | the decode window | 21/21; 12/12 (3 lab-model probes skipped) |
| `inference-t4c.json` | the prefill window too | 21/21 |
