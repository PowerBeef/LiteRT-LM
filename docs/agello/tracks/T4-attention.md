# T4: attention work bounded to the filled context

E4B's attention ends `runtime_bmm(Q, K) → SELECT_V2(mask, scores, −1e30) → SOFTMAX →
runtime_bmm(P, V)` (E4B-GRAPH.md). The two products stop at the filled length (`param_tensor`
entry 2), but the select and the softmax ran over the whole KV capacity. At a 32K window they took
about 60% of prefill GPU time (the select 23%, the softmax's elementwise pass 22%, its reduction
12%) and 12.7% of every decode step (TELEMETRY.md, GPU kernels).

## Step 1 (b1a): select and softmax bounded to the filled length (LiteRT `agello-next`)

A graph pass in the delegate, `FuseBoundedAttentionSoftmax`
(`ml_drift_delegate/delegate/composite/bounded_attention.{h,cc}`), rewrites each attention tail the
way ML Drift's own Gemma 4 builder and the delegate's SDPA fallback build theirs:

1. The select becomes a logits-first elementwise op with the infinite-float capping folded in, so
   ML Drift fuses it into the QK product's epilogue, which only writes the filled part (rounded up
   to 32 keys).
2. The softmax becomes a reduction (per row, the max and the sum's inverse) over
   `[0, params[2])`, through ML Drift's `SoftmaxRuntimeCheckDesc`.
3. The PV product reads the masked logits and computes the probabilities itself from that
   reduction (`src_exp`, a new `runtime_bmm` input), within its own bound.

No pass over the whole capacity remains. Keys past the filled length are masked (the causal mask
and the local window both end at the current position), so wherever a kernel rounds its bound up
to, it reads masked logits that contribute zero. The pass matches 42 tails in the decode and verify
graphs and 23 in each prefill graph (layers 0–23; layer 23's attention output is unused).
`AGELLO_BOUNDED_ATTENTION=0` turns it off; it's part of the program cache key.

Results (M6, E4B FP16, 2K prompt, 128 generated, same binary with the pass off and on, ABAB,
1 warm-up and 3 repetitions per arm; raw records in `results/mac/`):

| | Window 8,192 | Window 32,768 |
| --- | --- | --- |
| Prefill | 2,218 → 3,050 tok/s, +37.5% [+37.1, +39.8] | 1,135 → 2,726 tok/s, +140% [+135, +142] |
| Time to first token | −27% | −58% (1.8 → 0.8 s) |
| Decode | 48.3 → 49.8, +3.2% [+2.9, +3.8] | 43.1 → 49.2, +14.2% [+13.9, +14.3] |
| Decode with MTP | 155.0 → 156.9, +1.3% [+0.8, +1.7] | 134.6 → 154.1, +14.5% [+14.4, +14.8] |

Greedy replies are identical in all four configurations. Exit criterion (a 2K prompt at 32K within
10% of 8K): 2,726 against 3,050 tok/s, 10.6% slower, so just short; prefill at 32K is now faster
than it was at 8K before.

Fidelity, the digit-copy suites at a 32K window (`short`, `long`, `cross4096`, `xl`: prompts of
237 to 29,960 tokens): 88/88 copies exact with the pass off and on, and all 83 cases' replies
identical. The same run shows the gain across prompt lengths: prefill +102 to +110% for 237-token
prompts, +139% at 2K, +107% at 12K, +81% at 24K and +71% at 30K (945 → 1,616 tok/s: 18.5 s
instead of 31.7 s); +127% over all 83 cases (geometric mean), decode +13.9%. The gain shrinks as
the prompt fills the window, since the bound then covers more of it.

## In Agello (`calibrate`, MTP on, a prompt filling a third of the window)

The same binary with the pass off and on (`TEST_RUNNER_AGELLO_BOUNDED_ATTENTION=0`), single runs:

| Window | Prefill tok/s, off / on | TTFT s | Decode tok/s |
| --- | --- | --- | --- |
| 4,096 | 2,172 / 2,447 (+13%) | 0.59 / 0.53 | 78.7 / 83.5 |
| 8,192 | 2,215 / 2,939 (+33%) | 1.04 / 0.78 | 72.8 / 76.0 |
| 16,384 | 1,610 / 2,678 (+66%) | 2.73 / 1.64 | 74.9 / 75.5 |
| 32,768 | 1,099 / 2,347 (+114%) | 7.83 / 3.67 | 64.2 / 65.4 |

The first calibration after the cache key changed showed 4K slower (1,304 tok/s): one-off stalls
of about 430 ms in the CPU embedders' first inferences, which the pass doesn't touch (cold caches;
the bench saw nothing of the kind, warm-up runs included). Run after the `off` arm, the `on` arm
has the same embedding time as `off`. Agello's suites on this build: inference 21/21, probes 12/12
(3 lab-model probes skipped), thinking 2/2 (its margin test, which failed with both builds before,
passes), files 3/3; no error anomalies. Unit tests: `bounded_attention_test` (3 cases).

## Next

- b1b: the local layers (35 of 42) need only the last 512 keys: a window start bound for the QK
  product, the reduction and the PV product, from the position (no model change).
- b1c: FlashDecode for INT8 KV with GQA 4:1, and flash prefill for head dimensions 256/512.
- iPhone run before shipping: `inference`, `probes` (10, 12), `calibrate`, `thinking`.
