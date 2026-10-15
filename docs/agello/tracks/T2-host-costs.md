# T2: host costs per token

The decode loop waits for the host between GPU runs (TELEMETRY.md: sampling 0.50 ms, parameter
fill 0.24 ms and embedding lookups 0.14 ms per token on the M6), so host work per step is time
the GPU sits idle. This track removes what the telemetry counters showed to be repeated for
nothing.

Measured with agello-bench `suites/thinking-mask.json` (a 700-token prompt, 256 tokens, greedy,
8K window), MTP off and on, 1 warm-up and 3 repetitions; raw records in
`results/mac/2026-10-15/bench/t2-think-*.jsonl` (`pre`: build `3c34537772ce`; `post`: the mask
skip and the per-step changes; `post2`: plus `GetVocabSize`).

## P4.1 Thinking budget (`agello-next` `ef197a04`)

Think mode wraps every decode in a `ThinkingBudgetConstraint`. While the model thinks within its
budget, the constraint's mask allows every token, yet the mask runner applied it to all 262,144
logits each step (on device buffers, as a mask graph). Masks made by `CreateAllAllowed` now carry
a flag, and the runner skips one that covers the logits' vocabulary; one made for a smaller
vocabulary still disallows the rest (an existing test caught that case).

| Decode tok/s, median of 3 | Plain | Think (budget 4,096) |
| --- | --- | --- |
| `pre`, MTP off | 49.2 | 49.2 |
| `post`, MTP off | 49.4 | 49.2 |
| `pre`, MTP on | 159.6 | 104.9 |
| `post`, MTP on | 161.2 | 106.6 |

On the M6 the mask cost too little to measure: Think mode was already within 0.3% of plain decode
without MTP (the plan's exit criterion is 3%). With MTP, Think mode is 34% slower because the
drafter is right less often on thought text (158 of 294 drafts accepted, against 193 of 195 on the
filler the speed prompt repeats), not because of the mask. The skip matters more on the iPhone's
slower CPU; it needs the iPhone `thinking` suite to show it.

## P4.2 Per-step allocations and copies (`e43f9a8d`, `5e946049`)

- **Token history.** Each plain decode step copied the whole token history for the attention mask
  fill. The fill reads it only to find vision placeholders among the queries, and a decode step's
  query is a sampled token. The decode fill now passes no history.
- **Sampled ids.** Each plain step allocated a tensor buffer for the sampled ids; it's now kept for
  the next step of the same shape.
- **Vocabulary size.** Every decode task (every reply) built a string for each of the 262,144
  tokens to count them (`tokenizer.GetTokens().size()`); `GetVocabSize()` returns the same count.

| Counters per 256-token reply (MTP off) | `pre` | `post` |
| --- | --- | --- |
| `decode_allocations` | 256 | 0 |
| `token_history_copies` | 258 | 2 (prefill) |

Decode 49.2 → 50.2 tok/s (MTP off) and 159.6 → 162.2 (MTP on) between `pre` and `post2`. Those are
separate runs, not an A/B in one binary, so up to about 1% of it can be thermal drift. Time to
first token for the 700-token prompt: 310 → 307 ms, within noise.

## Not done

The plan's remaining T2 items: a text-only stream callback with a single parse in Swift, caching
the rendered template, per-step maps and handles in `BindTensorsAndRunDecode`, and the worker
thread's QoS. With the host already under 1 ms per token on the M6, these wait for an iPhone
profile that shows them.
