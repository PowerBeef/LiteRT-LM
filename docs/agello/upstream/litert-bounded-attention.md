# Draft pull requests (LiteRT + ML Drift): attention bounded to the filled KV length and the local window

Two PRs that go together: the delegate pass in LiteRT, and the kernel bounds it needs in ML
Drift. Track note: `docs/agello/tracks/T4-attention.md`.

## LiteRT PR: "ML Drift delegate: bound runtime_bmm attention tails to the filled length and the local window"

Models exported with `odml.runtime_bmm` (Gemma 4) end each attention with
`runtime_bmm(Q, K) → SELECT_V2(mask, scores, c) [→ MAXIMUM → MINIMUM] → SOFTMAX → runtime_bmm(P, V)`.
The two products stop at the filled length (`param_tensor`), but the select and the softmax run
over the whole KV capacity, and the parser drops `is_global`, so local layers read every position
instead of their sliding window. At a 32K window, the select and softmax took about 60% of prefill
GPU time and 12.7% of every decode step (Gemma 4 E4B, M6 Mac).

`FuseBoundedAttentionSoftmax` (`ml_drift_delegate/delegate/composite/bounded_attention.{h,cc}`,
registered with the custom transformations) rewrites each tail the way ML Drift's own Gemma 4
builder builds it:

1. The select becomes a logits-first elementwise op with the infinite-float capping folded in, so
   ML Drift fuses it into the QK product's epilogue, which only writes the filled part.
2. The softmax becomes a reduction (the max and the sum's inverse) bounded by the same
   `param_tensor` entry.
3. The PV product computes the probabilities itself from that reduction (`src_exp`, a new
   `runtime_bmm` input), within its own bound.
4. Local layers (`is_global = false`) also get a start bound: `FindSlidingWindow` reads W from the
   graph's `LOGICAL_AND(GREATER_EQUAL, LESS)` masks with int32 constant keys (W = 512 for Gemma 4)
   and keeps only keys in (p − W, p], from the window's start rounded down to 32.

Keys outside the bounds are masked, so wherever a kernel rounds a bound, it reads masked logits
that contribute zero. `AGELLO_BOUNDED_ATTENTION=0` and `AGELLO_BOUNDED_WINDOW=0` turn the pass and
the window off for A/B runs; both are part of the program cache key. Unit tests:
`bounded_attention_test` (7 cases).

Results (Gemma 4 E4B, FP16, M6 Mac, ABAB with the pass off and on in one binary):

| | 8K window | 32K window |
| --- | --- | --- |
| Prefill, 2K prompt | +37.5% (2,218 → 3,050 tok/s) | +140% (1,135 → 2,726) |
| Decode, 2K prompt | +3.2% | +14.2% |
| Decode with MTP | +1.3% | +14.5% |
| Decode, 30K filled (window on top) | | +30% (33.1 → 43.1 tok/s) |
| Prefill, 30K prompt (window on top) | | +21% (1,701 → 2,061 tok/s) |

Digit-copy fidelity suites (237 to 29,960-token prompts): 88/88 copies exact with and without the
pass, and every reply identical.

Commits on `agello-next` of https://github.com/PowerBeef/LiteRT: `ede28c641`, `aa05bd2d1`,
`313863175`.

## ML Drift PR: "Kernels: a start bound for sliding-window attention"

The window needs a start next to the existing end bound:

- `ConvRuntimeCheckDesc` and `SoftmaxRuntimeCheckDesc` gain `window_pos_index`, `window_size`,
  `dst_window` and `src_window`. A new `RuntimeWindowStartSlice()` computes
  `max(0, params[idx] − W + 1)` rounded down to 32 channels.
- The FC kernel skips the QK reduction below the start and starts PV's source loop there; the
  softmax and softmax-1x1 reductions start at `START_SLICE`.
- `conv_apple_mpp` does the same for prefill: the PV K-loop starts at the window, with the weight
  offset advanced, and QK runs zero K iterations for tiles below it.

Commits on `agello-next` of https://github.com/PowerBeef/ml-drift: `6ae7c68`, `2184a7f`.
