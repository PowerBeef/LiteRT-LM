# Draft issue (LiteRT-LM): host work repeated every decode step

**Title:** Decode: per-step host work that can go (token history copies, all-allowed masks,
vocabulary strings, buffers)

**Body:**

Decoding on the GPU waits for the host between runs, so host work per step is GPU idle time. Our
telemetry counters on Gemma 4 E4B (M6 Mac, v0.18.0-era `main`) showed work repeated for nothing:

1. **Token history copies.** `LlmLiteRtCompiledModelExecutorBase` copies the whole token history
   (`GetCopyOfTokens`) for the attention-mask fill of every plain decode step. The fill reads it
   only to find vision placeholders among the queries (`ExecutorVisionData::kSpecialToken`), and a
   decode step's query is a sampled token, never one. Counter: 258 copies per 256-token reply,
   growing with the context; 2 (prefill) without them.
2. **A tensor buffer per step** for the sampled ids (`CreateTensorBuffer<int>` in the plain decode
   and the MTP priming step); it can be kept for the next step of the same shape. 256 allocations
   per reply; 0 after.
3. **All-allowed masks.** With thinking enabled, every decode runs under
   `ThinkingBudgetConstraint`, whose mask allows every token while the model thinks within its
   budget, and the mask runner still applies it to all 262,144 logits (or runs a mask graph on
   device buffers). A flag on masks from `BitmapLogitMask::CreateAllAllowed` lets the runner skip
   them when they cover the logits' vocabulary. A smaller-vocabulary mask still disallows the rest,
   as `BitmapWithNegativeVocabDisallowsEverything` expects.
4. **Vocabulary strings.** `tasks.cc` takes the vocabulary size from
   `tokenizer.GetTokens().size()` on every decode task, building 262,144 strings;
   `GetVocabSize()` returns the same count.
5. **Greedy sampling on FP16 logits** (MTP's drafter and verifier, every step): `TopPSampler`
   allocates a vector for the halves, converts all of them to floats, then searches the floats.
   Comparing the halves as integers (flip the bits of negatives, set the sign bit of positives)
   finds the same first maximum with no conversion.

On the M6 these are each within a few percent (the host is fast); the counters go to zero and
replies are identical. They matter more on phone CPUs. Patches (branch `agello-next` of
https://github.com/PowerBeef/agello-litert-lm): `ef197a04` (masks), `e43f9a8d` (history copies,
buffers), `5e946049` (vocabulary size), `fdd5bac7` (FP16 greedy picks), with unit tests for each.
