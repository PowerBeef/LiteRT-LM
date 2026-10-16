# iPhone results, 2026-10-15

iPhone 17 Pro (A19 Pro, Apple10, 12 GB, app limit 8.6 GB, iOS 27.0.1; some report headers read
6.4 GB, an early reading, see ../../../reference/facts.md#f-071), our build `2e54fbdb9ca9`
unless named otherwise. Read with ../../VALIDATION.md. Home-directory paths replaced by `~`.

| File | What |
| --- | --- |
| `agello/{inference,probes,files,thinking,memory,calibrate,web}-ours.json` | The overnight suites in order (calibrate ran hot, thermal state 2) |
| `agello/memory-ours-telemetry0.json` | The memory suite again with telemetry off |
| `agello/memory-google.json`, `agello/probe06-google.json` | Google's v0.18.0 the same night: memory suite, probe 6 alone |
| `agello/calibrate-cooled-{ours1,google,ours2}.json` | Calibrate after a warm-up and a 15-minute cool-down, ours / Google's / ours |
| `agello/probe06-ours-xnnpack-fix.json`, `agello/inference-ours-xnnpack-fix.json` | After the XNNPACK fix: probe 6 alone (build `ab729fd0b59d`), inference (build `c397eca597bc`, the same sizing) |
| `agello/{inference,probes,memory,calibrate}-sim-a17.json` | `AGELLO_SIMULATE_APPLE_GPU="Apple A17 Pro"` (Apple9 kernel paths), build `d7fafdca9b7c` |
| `agello/memory-sim-a17-test0{2,3}_*-fresh.json` | The 4.0 and 3.5 GB envelopes on Apple9 paths, each in a fresh launch |
