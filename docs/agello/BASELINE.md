# Baselines (M0.6) and first parity (M1.6)

Mac mini M6 (`Mac18,5`, GPU family Apple11, 12.7 GB Metal working set, 16 GB), macOS 27.0.1,
Xcode 27.0, plugged in, thermal state nominal throughout. Models from `~/Models/agello/`:
our split-RoPE E4B (`gemma-4-E4B-it-fp16-rope.litertlm`, rev `13d5aa4c`) unless noted.
Raw records: `tools/agello-bench/results/` (git-ignored) and Agello's `build/mac-harness*/`.

## Arms

| Arm | Build | Backend on the Mac | How it runs |
| --- | --- | --- | --- |
| A0 | Google v0.18.0 release (`CLiteRTLM_mac`, sha256 `9526db9e…` dylib) | WebGPU (Dawn) | agello-bench, macOS |
| A1 | Google v0.18.0 release (`CLiteRTLM` iOS slice) | Metal | Agello as an iPhone app on the Mac (`PLATFORM=mac`) |
| A2 | Google nightly 0.19.0-20261013 (`CLiteRTLM_mac`, `a4a7ab6b…`) | WebGPU | agello-bench, macOS |
| A3 | Google nightly 0.19.0-20261013 (iOS slice) | Metal | Agello on the Mac |
| N1 | Ours: `agello-next` (the nightly's source + tool-call patch + Metal accelerator linked statically), `//agello:CLiteRTLM` | Metal, natively on macOS (Mac slice) and in Agello (iOS slice) | agello-bench and Agello |

## Engine benchmark (`LiteRTLM.benchmark`, 1,024 prefill / 256 decode, T = 1,312, no MTP)

| Arm | Prefill tok/s | Decode tok/s | TTFT s | Runs |
| --- | --- | --- | --- | --- |
| A0 (WebGPU) | 959–961 | 48.2 | – | 2 + warm-ups, ABAB with N1 |
| A1 (Google Metal, iPhone app on the Mac) | 3,211 | 49.8 | 0.34 | 1 (Agello `test90_Benchmark`) |
| N1 (our Metal, native) | 3,363–3,433 | 49.4 | – | 2 + warm-ups, ABAB with A0 |

Prefill on Metal is 3.5× WebGPU's; decode is within 3%, so the Mac's decode is bandwidth-bound
on both backends at this context.

## Fidelity

- Digit copies, `suites/short.json` (237–1,639 prompt tokens, FP16): A0 15/15 exact, N1 15/15
  exact, N1's replies identical to A0's on all 15. agello-bench reproduces the #3913 research
  run of the same suite exactly (15/15 identical replies and prompt sizes).

## Agello's suites on the Mac

| Suite | A1 (Google v0.18.0 Metal) | N1 (ours) |
| --- | --- | --- |
| unit | 220, 1 skipped (as on the iPhone) | – |
| inference | 21/21 | running |
| files | 3/3 | – |
| thinking | 1/2: `test02_ReasoningStaysOutOfCache` | – |

`test02_ReasoningStaysOutOfCache` fails on margin, not behavior: on the M6 the first answer's
reasoning came out short (204 bytes), and the test's floor (second reply plus its reasoning at 4
bytes per token, 100 tokens) can't separate "kept" (expected ≥ 100) from "dropped" when the first
reasoning is that short (measured growth 139). It needs a length-independent check (see S8).

## Noise

Same-build ABAB runs of `suites/sweep-8k.json` (A0): prefill within 0.5%, decode within 2%
(compare.py noise band).
