# Returns: x-memory (C1b, Run C), memory.md and troubleshooting.md

## Contradictions resolved

- **C1** (iPhone 17 Pro app limit, 8.6 GB vs 6.4 GB): resolved in memory.md (#c1-86-gb-or-64-gb) with
  B3's claims APL-001, APL-004, APL-006 (and APL-005 for the unmeasured unentitled limit). Same
  reading, 8 GiB steady; 6 GiB in early readings of 6 of 36 runs.
- **C9** (E4B KV at 32K, 918 vs 940 MB): resolved in memory.md (#c9-918-mb-or-940-mb) with
  MOD-E4B-039, LM-EXE-003 and MOD-FAM-047: 32,000 positions = 917,504,000 bytes (918 MB); 940 MB is
  32,768 positions, never allocated; the exported 32,003-position tensors would be 917,590,016
  bytes, also 918 MB.

## Notes for other pages

- **litert-lm/engine-and-sessions.md (LM-ENG-066) and anything quoting calibration "peaks":** the
  calibration report's `footprintPeakGB` is not a peak. `CalibrationScenarioTests.swift:66` (Agello
  `93122b647a`) reads `MemoryProbe.footprintBytes()` once, right after the first reply. The real
  high-water mark is our telemetry's `memory.peak_footprint_mb` (`task_vm_info`
  `ledger_phys_footprint_peak`, [`agello/telemetry/agello/agello_telemetry.cc:676-690`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/agello/telemetry/agello/agello_telemetry.cc#L676-L690)),
  which is process-wide: E4B at 32K reached 4,809.4 MB in `calibrate-host-idle.json` (X-MEM-005),
  against the 3.851 GB "peak" field. LM-ENG-066's figures also include the test host's own engine
  (all reports before Agello `8bae20b`; X-MEM-011). Suggest "footprint after the reply" for that
  field wherever it's quoted.
- **apple-platform.md:** its envelope section defers footprints to memory.md; memory.md now has the
  idle-host step table (X-MEM-009) and the test-host explanation (X-MEM-011). Its Limits table could
  add "~1.2 GB more in reports before `8bae20b`: the test host's engine".
- **configuration.md (rows):**

  name | layer | default | where read | effect on Apple | version added | Agello sets it? | notes
  --- | --- | --- | --- | --- | --- | --- | ---
  `EngineCache` folder | Agello | `Library/Caches/com.patricedery.agello.litertlm/{kv-all-<id>,embedding-<id>}` (ours) or `{kv-<n>,embedding}` (Google's) | `Agello/Inference/EngineCache.swift:21-130` | one cache folder per runtime build; two context sizes kept with Google's binary | Agello | yes | X-MEM-014
  `prefill_preface_on_init` | `ConversationConfig::Builder` | false | `runtime/conversation/conversation.cc:437-438` | when false, nothing is prefilled at creation and `getTokenCount()` reads 0 until the first message | upstream | no (not exposed in C or Swift) | X-TRB-004

- **glossary.md:** "high-water mark" (`ledger_phys_footprint_peak`, process-wide, never lower than an
  earlier reading); "footprint after load" (Agello's reading after the one-token validation);
  "idle test host" (Agello `8bae20b` onward: the app shows nothing while it hosts tests).
- **facts.md candidates:** E4B footprint after load on the iPhone 17 Pro, our build, idle host:
  1.019 / 1.245 / 1.862 / 2.853 GB at 4K / 8K / 16K / 32K (X-MEM-005); Google's 1.095 / 0.988 /
  1.648 / 2.862 GB (X-MEM-006); 32K high-water mark 4.81 GB; first image +1.29 to +1.36 GB
  (X-MEM-009); kv-all folder 2.47 to 2.56 GB (OUR-023, X-MEM-012).
- **contributing.md / patches.md (upstream candidates, not posted):**
  - X-TRB-001: a section that fails to map is logged and skipped; `InitializeEmbeddingLookups`
    silently creates no per-layer lookup, the engine is created, and every prefill fails "No
    per-layer embeddings found in inputs.". Same in v0.18.0. A one-line report: fail engine creation
    when a file's per-layer embedder section exists but can't be loaded.
  - X-TRB-003: after a full KV cache, the next `Conversation` message completes with an empty reply
    and no error (default `return_error_on_max_tokens_reached` false), which callers can't tell from
    a real empty answer.
- **tools.md:** probe 14's folder sizes and probe 16 (nothing left after unload) are the two
  memory probes memory.md relies on; `calibrate` records `telemetry.memory` per window.

## For Agello (not a reference page; for the user)

- `CalibrationScenarioTests` names a single after-reply reading `footprintPeakGB`; a real peak is in
  `telemetry.memory.peak_footprint_mb` (our runtime only).
- `DeviceProfile.keepsConversationAcrossHelpers` counts 28,672 × 32,768 bytes at a 32K setting,
  22,020,096 bytes more than the 32,000 positions the runtime allocates (X-MEM-016); harmless
  (conservative).
- The first image costs about 1.3 GB on both runtimes and stays until the engine is reloaded
  (X-MEM-009); worth knowing for the 8 GB-iPhone envelope.

## Open questions

- Resident weight pages: how much of the 2.2 GB weight cache and the 1.0 GB CPU embedding tables is
  resident during a reply, and how iOS accounts `newBufferWithBytesNoCopy`-wrapped file pages (a
  `footprint`/`vmmap` capture on the iPhone would answer it).
- About 1.03 GB of footprint growth from 4K to 32K that the KV cache doesn't explain (X-MEM-007):
  which ML Drift intermediates; no telemetry of the memory plan exists.
- What the ~1.3 GB of the first image is (vision encoder weights in XNNPACK form, activations for
  2,520 patches, or other CPU buffers).
- The split of the `kv-all` folder between weight cache and program cache files; whether the drafter's
  weights are in it.
- Heap corruption on KV overflow (Agello's skill, shipped binary): not traced in the documented
  build's code; candidate paths (a multi-token MTP step near the end, a filter refill) unverified.
- MTP with JSON-schema constraints on the documented build: untested; the verifier applies the mask
  there (LM-EXE-024).
- The v0.18.0 conversation code's KV-full and cancel chains weren't traced end to end (the documented
  build's are: X-TRB-003, X-TRB-005).
