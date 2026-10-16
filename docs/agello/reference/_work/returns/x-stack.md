# Returns: x-stack (area C1a, Run C)

Pages written: `stack.md`, `tools.md`, `maintenance.md`. Claims: `_work/claims/x-stack.jsonl`
(X-STK-001 to -011, X-TLS-001 to -017; 19 tier A, all tagged; 9 tier B, untagged; no X-MNT claims,
see below). Everything else on the pages restates verified or corrected ledger claims, tagged with
their IDs (222 tags across the three pages, all checked against the ledger).

## Contradictions and corrections found

- **MD-MTL-022 vs APL-051 (argument buffers).** MD-MTL-022 (corrected) says the Metal path binds the
  argument buffer "with argument buffers, which LiteRT-LM requests on Apple". APL-051 (corrected)
  says only `SetCommonGpuOptions` (vision, audio and embedding executors) requests them and the main
  LLM executor's `CreateCompilationOptions` doesn't, so the Gemma decoder graphs don't use argument
  buffers. A grep at `7b872544` confirms APL-051: `SetUseMetalArgumentBuffers(true)` appears only in
  `SetCommonGpuOptions` (`litert_compiled_model_executor_utils.cc:1034`), which
  `audio_litert_compiled_model_executor.cc` and `embedding_litert_compiled_model_executor.cc` call,
  not `llm_executor_settings_utils.cc`. Resolution used in `stack.md`: the decoder doesn't use
  argument buffers; MD-MTL-022's parenthesis needs a `z-audit` correction (and `metal-backend.md`'s
  wording, if it repeats it).
- **`generated/cli-flags.md` "`litert_lm_main` and `litert_lm_advanced_main` share
  `shared_flags.cc`".** True of linking, misleading for use: `litert_lm_main` reads only seven flags
  (`--backend` default `gpu`, `--model_path`, `--input_prompt`, `--input_prompt_file`,
  `--visual_token_budget`, `--num_cpu_threads`, `--enable_ynnpack`); the other shared flags are
  accepted and ignored. `litert_lm_advanced_main` reads them and its own `--backend` defaults to
  `cpu`. Claim X-TLS-004. `gen_flags.py` could print which binary reads each flag.
- **C6, a fourth stale point.** agello-bench's README run-options sentence (at `85b1a776` and at the
  `agello` tip `61c03d8e`) omits `--no-benchmark-mode`, `--telemetry` and `--trace`, which the usage
  text has had since `85b1a776`. Claim X-TLS-007.
- **TELEMETRY.md graph dumps.** Restated from MD-MTL-034 on `tools.md`: graph dumps work at any level;
  TELEMETRY.md's level table (line 34) lists them under level 4, its own "GPU graphs" row says any
  level.
- **The brief's location of agello-bench.** The brief says agello-bench is in the agello-next
  worktree; it is on the fork's `agello` branch (the checkout at `/Users/patricedery/Coding
  projects/LiteRT-LM`, tip `61c03d8e`), and `tools/` in the agello-next worktree holds only `pytest/`.
  By default the bench builds against that `agello` checkout, i.e. Google's v0.18.0 binary
  (X-TLS-006); our arm is a `LITERTLM_PACKAGE` build against the agello-next worktree.
- **`pins.json` comment.** `remote_models_comment` names `gen_model_tables.py --remote`; the script has
  only `--only <key>...` and reads remote files on every run.

## Notes for other pages

- **README.md (Run C, later):** `stack.md` is the starting point for every reading path (it links each
  layer's page); `maintenance.md` is the page for whoever updates pins, models or measurements;
  `tools.md` for commands.
- **facts.md:** candidate fact: a plain E4B decode step in the documented build moves at least
  577,536 bytes between host and GPU-visible memory (53,248 in, 524,288 out), X-STK-010.
- **glossary.md:** terms used on these pages: *plain decode step* (one `decode` run), *MTP round*
  (three drafter runs and one `verify` run), *arm* (agello-bench: a LiteRT-LM checkout with its
  wrapper and binary), *verdict batch* (`v1`, `v2`, `x1`, `z-audit`; later names win the merge).
- **configuration.md:** rows to consider: `LITERTLM_PACKAGE` (agello-bench `Package.swift`, default
  `../..`), `AGELLO_BENCH_HOLD` and `AGELLO_BENCH_RELEASE_TRACE` (agello tip only, X-TLS-008), Agello's
  harness default `AGELLO_TELEMETRY=2` for tests (X-TLS-009); CLI flag rows should say which binary
  reads them (X-TLS-004).
- **swift-api.md / c-api.md:** per streamed chunk the Swift wrapper parses the same JSON text twice
  (`JSONSerialization`, then `jsonToMessage` parses it again), in both wrappers (X-STK-009); worth a
  line where streaming cost is discussed, and a possible upstream note.
- **conversation.md:** the hop from `litert_lm_conversation_send_message_stream` to
  `SendMessageAsync` and the per-chunk work of `CreateInternalCallback` are now claims (X-STK-003,
  -004, -008) if that page wants to tag them.
- **tools.md / maintenance.md readers:** `tools/agello-docs/` and the reference itself were untracked
  in the fork checkout on 2026-10-15. No claim can cite the scripts until they are committed, which is
  why `maintenance.md` and the `tools/agello-docs` section of `tools.md` carry no X-MNT/X-TLS claims
  about the scripts' behavior. Committing them (on `agello` or `agello-next`) would also give
  `patch_rope.py`-style tools a pinned home.

## Open questions

- Which sampler Google's v0.18.0 binary uses at run time on an iPhone (symbols only).
- The per-chunk cost of the layers above `Tasks` (`stream_callback` span) has no results file of its
  own.
- The CPU cost of the active Metal wait on the iPhone in Agello's configuration (benchmark mode on).
- `litert_lm_main` / `litert_lm_advanced_main` from our fork on macOS: not built or run; the static
  Metal link is read from BUILD files (X-TLS-003).
- Whether `x1` verdict batches have a defined meaning beyond "a further pass".
