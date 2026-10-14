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

## A float16-exact rotary path at every length (`--split`)

A constant shift only helps up to 4,096 positions: one float16 tensor can't
hold more than ~4,097 evenly spaced integers. `patch_rope.py --split` makes the
rotary phases exact for positions 0–32,767 with float16 activations, in both
`tf_lite_prefill_decode` and `tf_lite_mtp_drafter`. It splits the position as
`2048·h + l`:

- **l (0–2047, exact in float16)** goes through the existing MUL / SIN / COS.
- **The phase of `2048·h`** comes from a precomputed 16 × F table, selected
  with `onehot(h)`.
- **Angle addition** combines the two, using only values in [−1, 1].

Results on the Mac (WebGPU delegate):

| Check | Result |
| --- | --- |
| FP32: patched text vs original text | identical, 40/40 |
| FP16, prompts of 237–4,449 tokens (`short`, `long`, `cross4096`) | 70/70 exact (original: 15/70) |
| FP16, prompts of 6K–30K tokens (`xl`, `maxNumTokens` 32768) | 18/18 exact, **identical to original FP32** in 18/18 (original FP16: 0/18) |
| CPU, long prompts | 3/3 exact |
| Speed (no MTP) | prefill and decode within ~1% of the original FP16. Original FP32 prefill is ~35% slower. |
| With speculative decoding (`MTP=1`) | 12/12 exact, identical to original FP32 + MTP. 87–120 tok/s against 37–89 for original FP32 + MTP (2.35× faster at 16K). |

The first MTP group (703 tokens) includes each engine's warm-up run. Repeated
after warm-up, the split model with the drafter decodes at 123.7 tok/s, and
with only the main section patched at 124.4.

On the iPhone 17 Pro (Metal accelerator, probe 12, `maxNumTokens` 32768):

| Check | Split, FP16 | Original, FP32 |
| --- | --- | --- |
| Exact copies, 0.7K–30K tokens | 14/14, text identical to original FP32 in 14/14 | 14/14 |
| Original, FP16 | n/a | 0/3 (2K, 8K, 30K) |
| FP32: split text vs original text | identical, 4/4 | n/a |
| Prefill tok/s (2K · 8K · 30K) | 512 · 385 · 288 | 243 · 229 · 154 |
| Decode tok/s (2K · 8K · 30K) | 22.2 · 16.1 · 11.2 | 14.2 · 13.7 · 10.4 |
| With MTP: decode tok/s (2K · 6K · 16K) | 38.6 · 40.5 · 34.7, 6/6 exact, identical to FP32 + MTP | 25.2 · 24.2 · 17.1 |
| Footprint after load | 2.04 GB | 2.76 GB |

At the same lengths the original FP16 runs at the same speed as the split
model (8K: 378 / 16.0 tok/s; 30K: 301 / 11.7). The patch costs nothing; it
removes the errors.

Two findings on the WebGPU delegate shaped the design:
- **FULLY_CONNECTED with float32 weights** computed wrong results, even with FP32 activations.
- **REDUCE_MAX over an inner axis**, or over a 3-D tensor, effectively returned
  the first element along the reduced axis.

So every new tensor is 4-D and every reduction runs over the last axis, like the
model's own REDUCE_MAX ops; tables are selected with MUL + REDUCE_MAX
(`onehot · (table + 2)`, then −2).

```sh
python patch_rope.py models/gemma-4-E4B-it.litertlm --split models/split.litertlm
./run_arm.sh split gpu fp16 long split
./run_arm.sh split gpu fp16 xl   split 32768
MTP=1 ./run_arm.sh split gpu fp16 mtp mtp-split 32768
```

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
python patch_rope.py models/gemma-4-E4B-it.litertlm --split models/split.litertlm

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
| `patch_rope.py` | Shifts the rotary position (`<shift>`, `--int`) or makes it float16-exact (`--split`) in `tf_lite_prefill_decode` and `tf_lite_mtp_drafter` |
| `runner/` | macOS copy-fidelity runner (Swift, this repository's package) |
| `cases/` | Prompt cases: `baseline` (the iPhone probe's cases), `short`, `long`, `long3`, `cross4096`, `xl` (6K–30K tokens), `speed`, `mtp` |
| `results/mac/` | One JSON line per reply: `<cases>__<model>__<backend>-<precision>.jsonl` |
| `results/iphone/` | iRTChat probe results: probe 10 (position boundary), probe 11 (INT32 shift) |
| `summarize.py` | Prints the tables from the results (no dependencies) |
