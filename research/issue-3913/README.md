# LiteRT-LM #3913: float16 rotary positions

Reproduction for [google-ai-edge/LiteRT-LM#3913](https://github.com/google-ai-edge/LiteRT-LM/issues/3913)
(GPU, default float16 activations: Gemma 4 duplicates digits in long contexts).

## Finding

In `gemma-4-E4B-it.litertlm`, the rotary angles are computed at runtime from a
float copy of the INT32 position:

```text
input_pos (INT32) → RESHAPE → CAST → …/maybe_rope/convert_element_type (FLOAT32)
  → MUL(…/maybe_rope/div) → SIN / COS
```

With float16 activations, that float position can't represent every integer
above 2048 (2049 → 2048, {2051, 2052, 2053} → 2052). Neighbouring tokens get
identical rotary phases, and verbatim copies stutter.

Shifting only the rotary position, which changes nothing in exact arithmetic,
moves the failures:

| Patch (rotary path only) | Prompt tokens | Original, FP16 | Patched, FP16 | Patched, FP32 |
| --- | --- | --- | --- | --- |
| +2048, float, after the CAST | 237–1,639 | 15/15 | **0/15** | 15/15 |
| −2048, INT32, before the CAST | 2,094–3,983 | 0/25 | **25/25** (Mac and iPhone) | 25/25 |
| −2048, float, after the CAST | 2,094–3,983 | 0/25 | 0/25 | 25/25 |

With FP32 activations, every patched model's text is identical to the
original's (65/65), so the patch itself is neutral. `python3 summarize.py`
prints every number from the raw results.

## Environment

- LiteRT-LM v0.18.0: this branch is based on the `v0.18.0` tag, and the
  runner uses its Swift package and released `CLiteRTLM` binaries.
- Model: `litert-community/gemma-4-E4B-it-litert-lm` at revision `2eee7ac3`,
  `gemma-4-E4B-it.litertlm`, sha256
  `0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0`.
- Mac: Apple M6, macOS 27.0.1. GPU runs on the WebGPU delegate (Metal adapter).
- iPhone 17 Pro, iOS 27.0.1. GPU runs on the Metal accelerator. The iPhone
  runs use the same prompts, in
  [iRTChat's probes](https://github.com/PowerBeef/iRTChat/blob/055585262d3b156ca6bacb52ee7cd14360d59ae5/iRTChatDeviceTests/ProbeScenarioTests.swift)
  (`test10_PositionBoundary`, `test11_ShiftedRotaryPositions`).

## Method

- **Task:** copy a 48-digit reference out of digit-free filler.
- **Decoding:** greedy (`topK: 1, temperature: 0`), speculative decoding off,
  no system prompt.
- **Positions:** each digit is one token (confirmed by decode counts), so the
  digit at index `k` is predicted at position `prefill - 1 + k`.

## Reproduce on a Mac

```sh
# 1. Model
mkdir -p models
curl -L -o models/gemma-4-E4B-it.litertlm \
  https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/2eee7ac325f20eb8c9ac1d0e972f7c84663062da/gemma-4-E4B-it.litertlm

# 2. Patched models (about 1 minute each; ~3.7 GB each)
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python patch_rope.py models/gemma-4-E4B-it.litertlm  2048 models/float-plus2048.litertlm
python patch_rope.py models/gemma-4-E4B-it.litertlm -2048 models/float-minus2048.litertlm
python patch_rope.py models/gemma-4-E4B-it.litertlm -2048 models/int-minus2048.litertlm --int

# 3. Runner
(cd runner && swift build -c release)

# 4. Arms: ./run_arm.sh <model-stem> <gpu|cpu> <fp16|fp32> <cases> <label>
./run_arm.sh gemma-4-E4B-it gpu fp16 short original
./run_arm.sh float-plus2048 gpu fp16 short float-plus2048
./run_arm.sh float-plus2048 gpu fp32 short float-plus2048
./run_arm.sh gemma-4-E4B-it gpu fp16 long  original
./run_arm.sh int-minus2048  gpu fp16 long  int-minus2048
./run_arm.sh int-minus2048  gpu fp32 long  int-minus2048
./run_arm.sh gemma-4-E4B-it gpu fp32 long  original
python3 summarize.py
```

## Files

| Path | Contents |
| --- | --- |
| `patch_rope.py` | Shifts the rotary position in every signature of the `tf_lite_prefill_decode` section |
| `runner/` | macOS copy-fidelity runner (Swift, this repository's package) |
| `cases/` | Prompt cases: `baseline` (the iPhone probe's cases), `short`, `long`, `cross4096` |
| `results/mac/` | One JSON line per reply: `<cases>__<model>__<backend>-<precision>.jsonl` |
| `results/iphone/` | iRTChat probe results: probe 10 (position boundary), probe 11 (INT32 shift) |
| `summarize.py` | Prints the tables from the results (no dependencies) |
