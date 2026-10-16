# Returns: x-facts (C2, Run C): facts.md, glossary.md, _work/contradictions.md

## New claims

Five, all tier A, in `_work/claims/x-facts.jsonl`:

- X-FCT-001: Google's E4B iPhone 17 Pro card figures (1,189 prefill, 25.1 decode tok/s, 0.9 s TTFT; 1,024/256 tokens, 2,048 context; the card's speculative-decoding table is Android-only). Read at revision `2eee7ac3`.
- X-FCT-002: our build `2e54fbdb9ca9` vs Google's v0.18.0 in the cooled calibration: 1.84x / 2.20x at 32K, 1.04x / 1.08x at 4K (recomputed from the unrounded `decodeTokPerSec` values and from LM-VER-025's table).
- X-FCT-003: Agello's two downloads total 3,825,352,704 bytes (3.83 GB).
- X-FCT-004: Agello's keep-across-helper threshold at 8,192 / 16,384 / 32,768 tokens (634,881,024 / 869,762,048 / 1,339,524,096 bytes).
- X-CON-001: C15, AUDIT.md's line numbers are v0.18.0 (`git show agello:<path>`).

## Notes for other pages

- **README.md (Run C, later):** facts are addressable as `facts.md#f-NNN` (HTML anchors on every row); the glossary's first table ("Words that mean two things") is the best entry point for writers of user docs. The contradictions dossier is `_work/contradictions.md` (C1-C17, N1-N44).
- **memory.md:** F-030 (total download), F-063 (keep threshold at three sizes, X-FCT-004) and F-077 (envelope) are ready to link; facts.md links memory.md from the KV and memory sections.
- **versions.md:** F-109 (X-FCT-002) gives the 32K ratio; Agello README:102 rounds Google's 25.1 to 25 (fine, but say "model card, 2026-10-15").
- **litert-lm/conversation.md:** the mod-family returns' rewording of "so the model continues its turn" (N17) is still pending in that page.
- **maintenance.md:** on each re-pin, re-read the vendor-claim and project-record rows of facts.md (F-005, F-015, F-105, F-106, F-127, F-133); after the rebase that drops the XNNPACK patch, update F-129 and F-085.
- **Agello (for the user, via Run D):** additional C16 location found: `Agello/Inference/DeviceProfile.swift:60-62` comment ("Calibrated on iPhone 17 Pro … 43 → 38 → 34 → 27 tok/s") describes Google's v0.18.0 binary; `DeviceProfile.swift:42-43` repeats the "wipes" wording of N23.
- **Leads not checked by any claim** (left as leads in the dossier's Limits): Agello README "~0.4 s" TTFT and "~1.2-2.8 GB" footprint; ENGINEERING "0.7 GB less memory" (the #3913 report says about 0.45 GB, MOD-E4B-082; they may measure different things) and "10.25 GB" cache growth.

## Open questions

- Whether Google's iPhone 17 Pro figures (F-082, F-106) were measured on the current file revision; the card doesn't name it.
- The full device-scenario count from one harness run (49 declared at `93122b6`), if Agello's README is to state it (C7).
- N22, N40, N41 in the dossier stay open (EmbeddingGemma at FP16, the `.bazelrc` comment, drafter key scales).
