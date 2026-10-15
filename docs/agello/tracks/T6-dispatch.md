# T6: dispatch overhead

The plan: measure the GPU's idle time between kernels and the encode time per decode step, and
act (indirect command buffers, pre-encoding the next step, fusing elementwise chains) only if it's
worth it.

## Measurement (M6, E4B FP16, 700-token prompt, 8K window, telemetry level 2)

Command-buffer GPU time (`GPUEndTime − GPUStartTime`, summed per run) against the run's wall time
and the decode step's; `results/mac/2026-10-15/bench/t6-{plain,mtp}.jsonl`:

| | Without MTP (per token) | With MTP (per 4-token round) |
| --- | --- | --- |
| Step or round | 20.57 ms | 25.27 ms |
| GPU time | 20.26 ms (9 command buffers) | 23.85 ms (verify 21.94, drafter 1.91) |
| Run wall time over GPU time (bind, encode, commit, wait) | 0.24 ms (1.2%) | 0.69 ms (2.7%) |
| Host work outside the runs | 0.07 ms (0.3%) | 0.73 ms (2.9%) |
| GPU busy | 98.5% | 94.4% |

Command buffers are committed as they're encoded (LiteRT `77ed9cab`), so encoding overlaps the
GPU's work; what remains is the tail after the last commit and the wait's wake-up.

What this level doesn't see: gaps between dispatches inside a command buffer count as GPU time.
That doesn't change the decision for pre-encoding or indirect command buffers, which cut CPU
encode time: the GPU only waits on encoding between command buffers, the 0.24 ms above. Gaps
between dispatches would be kernel fusion's to recover (the elementwise chains around the norms),
and measuring them needs the shader timeline (Metal System Trace with GPU counters) or
telemetry level 4, whose one-encoder-per-kernel timing changes what it measures.

## Decision

Not worth acting on now: pre-encoding or indirect command buffers could recover at most the
1.2% without MTP. With MTP, the 5.6% outside GPU time is split between the drafter's host steps
(input concatenation, embedding lookup, greedy pick) and run overheads; drafting on the GPU (T12)
is the change that would address it, not encoding. Revisit with an iPhone Metal System Trace
(shader timeline on), where encode and wake-up costs are larger relative to the GPU.
