# Spot audits

## After Run A (2026-10-15)

Twenty claims drawn at random from the Run A ledger (one tier-A and one tier-B claim per group, all
"verified" or "corrected"), re-checked by two fresh agents against the pinned sources without seeing
the pages or the ledger.

| Result | Claims |
| --- | --- |
| Holds | 18: LM-ENG-057, RT-CPU-015, LM-VER-026, LM-EXE-073, LM-CAPI-044, MD-MTL-021, LM-CNV-017, MD-MTL-042, LM-OTH-004, MD-KRN-071, RT-DLG-067, LM-CMP-004, LM-EXE-026, RT-CPU-031, LM-CMP-008, RT-DLG-066, LM-CNV-028, LM-CAPI-032 |
| Imprecise | 2: LM-ENG-062 (the first-sampler conclusion ignored MTP's own greedy samplers), MD-KRN-032 (FC work-group sizing left out batch, bit-width and MoE cases) |
| Wrong | 0 |

The plan's threshold (more than one wrong claim re-verifies the area) isn't met. Both imprecise
claims are corrected in the ledger (verdict files `<group>.z-audit.jsonl`) and in the pages. The
lesson for later runs: verifiers accepted claims that left out a condition changing the result;
Run B and C verifiers are told to correct those.

## Final spot audit (2026-10-15, during Run C)

Twenty more claims (two tier-A claims per Run B group, one per Run A group not audited before),
re-checked the same way.

| Result | Claims |
| --- | --- |
| Holds | 18: APL-067, LM-VER-001, MOD-E2B-046, MD-MTL-004, RT-RUN-018, MOD-EMB-020, LM-CNV-070, OUR-017, MOD-FAM-005, LM-EXE-015, RT-DLG-013, APL-032, MOD-FAM-046, LM-SWIFT-033, MOD-EMB-074, LM-FMT-012, MOD-E4B-018, MOD-E4B-005 |
| Imprecise | 2: MOD-E2B-006 (the parameter count left out the checkpoint's unused k/v projections of the shared layers; both totals still round to 2.3 B), OUR-015 (left out two upstream ML Drift commits that also rewrite `fully_connected.cc`) |
| Wrong | 0 |

Corrected in the ledger (`<group>.z-audit.jsonl`) and the pages. Across both audits: 36 of 40 hold,
4 imprecise, 0 wrong. Every imprecise claim left out a case; none stated something false.

## After Run C (2026-10-15)

- APL-051: corrected again. Run B's correction said the Gemma decoder doesn't request argument
  buffers; the code at `7b872544` (`llm_executor_settings_utils.cc:163`) and at v0.18.0 (`:162`) calls
  `SetCommonGpuOptions`, which requests them under `__APPLE__`. Found by the x-config writer
  (X-CFG-009) and the QA reviewer; checked by hand. MD-MTL-022's corrected text was already right.
