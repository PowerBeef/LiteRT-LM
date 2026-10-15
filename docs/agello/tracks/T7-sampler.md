# T7: the sampler

Google's release binaries sample on the GPU with a closed-source Metal sampler
(`LiteRtTopKMetalSampler_*`); our source build falls back to the CPU sampler (`TopPSampler`). The
plan was an open Metal sampler behind the same C API (`default_static_gpu_samplers`). Before
writing one, the question was how much sampling still costs once T5 made logits reads direct.

## Where a decode step's time goes now (M6, E4B FP16, 700-token prompt, 8K window)

Telemetry level 2, build `f057206e0b1e`; `results/mac/2026-10-15/bench/t6-{plain,mtp}.jsonl`:

| Per step (no MTP) | ms | Share |
| --- | --- | --- |
| Decode step | 20.57 | 100% |
| Model run (bind, encode, GPU, wait) | 20.50 | 99.6% |
| GPU time (9 command buffers) | 20.26 | 98.5% |
| Sampling (greedy, CPU) | 0.044 | 0.2% |
| Embedding lookups (2) | 0.028 | 0.1% |

| Per MTP round (4 tokens) | ms | Share |
| --- | --- | --- |
| Round | 25.27 | 100% |
| Verify run / its GPU time | 22.23 / 21.94 | 88% / 87% |
| 3 drafter runs / their GPU time | 2.31 / 1.91 | 9% / 8% |
| GPU busy | 23.85 | 94.4% |

MTP's drafter and verifier always sample greedily (`CreateGreedySampler`). Without MTP, Agello's
chat presets sample top-k/top-p on the CPU, which this run doesn't measure: the 0.5 ms per token
in TELEMETRY.md was a greedy pick before T5 made logits reads direct.

## P6: the cheap part (`agello-next` `fdd5bac7`)

Greedy picks over FP16 logits no longer allocate, convert the 262,144 halves to floats, and search
the floats: the halves are compared as integers. Outputs identical; on the M6 within noise (MTP
159.7 → 159.6 tok/s, Think with MTP 104.9 → 105.3; `p6-*.jsonl`, 1 warm-up and 5 repetitions).

## Decision

No open Metal sampler for now. On the M6 a GPU sampler could recover at most the sampling share,
0.2% of a step for greedy picks (all of MTP, Agello's default). That isn't worth a Metal top-k,
top-p and RNG implementation that must match the CPU sampler's results. Revisit if a profile of
top-k sampling without MTP, or an iPhone profile, shows sampling above 3% of a step, or for T12
(drafting on the GPU), where a GPU argmax would let the drafter loop stay on the GPU.
