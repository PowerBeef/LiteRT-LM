# Style and evidence rules for docs/agello/reference

Every writer, verifier and editor follows this file. The goal: a reference that is right, says how
it knows, and can be re-checked when the code moves.

## What the reference documents

| Thing | Exactly |
| --- | --- |
| LiteRT-LM | `PowerBeef/agello-litert-lm` branch `agello-next` at `7b872544` (Google's `0.19.0-nightly.20261013` `df6df5b8` plus 29 commits of ours). Local checkout: `/Users/patricedery/Coding projects/agello-arms/next/LiteRT-LM` |
| LiteRT | `PowerBeef/LiteRT` `agello-next` at `362759838` (Google's `ca21dc99d` plus 12 commits). `/Users/patricedery/Coding projects/LiteRT` |
| ML Drift | `PowerBeef/ml-drift` `agello-next` at `15fa9e2` (Google's `1b18771` plus 8 commits). `/Users/patricedery/Coding projects/ml-drift` |
| What Agello ships today | The Swift wrapper at `PowerBeef/agello-litert-lm` `85b1a776` (v0.18.0 plus two Swift patches) with Google's prebuilt v0.18.0 `CLiteRTLM.xcframework`. Source of the wrapper: `/Users/patricedery/Coding projects/LiteRT-LM` (branch `agello`; its other source files are v0.18.0) |
| Models | `tools/agello-docs/pins.json` "models", "remote_models", "hf_repos" |

`tools/agello-docs/pins.json` is the source of truth for pins. When a statement differs between the
documented build and what Agello ships, say which one it is about.

## Page template

```markdown
# <Title>

> **Scope:** ... **Out of scope:** ... (with links to the pages that cover it)
> **Pinned to:** LiteRT-LM `7b872544`, LiteRT `362759838`, ML Drift `15fa9e2` (only those that apply)
> **Applies to:** Apple (iOS, macOS) / all platforms. **Last verified:** 2026-10-DD

## Summary            ≤ 10 lines: what it is, what matters for us
## Concepts           terms, data structures, invariants
## How it works       call paths with citations; mermaid sequence or flow diagrams where they help
## On Apple           what's active on iOS/macOS (Metal, static linking, unified memory, ...)
## Settings           the knobs that change it (rows live in configuration.md; link them)
## Limits and failure modes
## Where in the code  table: concept | symbol | citation
## Open questions     anything not verified, labeled as such
```

Omit a section that has nothing to say rather than padding it. Keep pages under ~900 lines.

## Citations

- Code citations are permalinks made with `tools/agello-docs/cite.py`, never typed by hand:
  `python3 cite.py litert-lm runtime/core/tasks.cc --symbol "Decode("` → ``[`runtime/core/tasks.cc:614`](https://github.com/...#L614)``.
  Files our commits don't change link to Google's repository at the upstream base; files we change
  link to our fork at the pin. `cite.py --json` prints the ledger source record.
- Cite the narrowest range that shows the claim (one function, one branch, one table row).
- Links to other reference pages are relative (`../litert/runtime.md#tensor-buffers`).
- Generated tables (`generated/`) are evidence and can be cited by relative link. Quote their key
  rows; don't copy whole tables.
- Web sources: URL, publisher, version (HF revision, arXiv vN, release tag, or "unversioned"),
  retrieved date (2026-10-15), and a quote of at most 25 words in the ledger. Unversioned pages
  (ai.google.dev, blogs) are marked *volatile*.
- Never reproduce long passages of Google's or Apple's docs. Summarize in your own words; quote at
  most a sentence, attributed.

## Claims and the ledger

Every statement of fact a reader could rely on is a claim. Writers return claims; the main loop stores
them in `ledger/<area>.jsonl`. Put a tag after the sentence that states a tier-A claim:
`... uses 28,672 bytes per token of context. <!-- c:MOD-E4B-007 -->`

```json
{ "id": "LM-ENG-042", "page": "litert-lm/engine-and-sessions.md", "anchor": "how-it-works",
  "text": "the one-sentence claim", "tier": "A|B",
  "kind": "code-structure|code-behavior|apple-path|number|model-structure|measurement|external|process",
  "sources": [ {"type": "code", ...cite.py --json record...},
               {"type": "web", "url": "", "publisher": "", "version": "", "retrieved": "2026-10-15", "quote": ""},
               {"type": "empirical", "path": "generated/...", "detail": "which row"},
               {"type": "results", "path": "docs/agello/results/...", "conditions": ""} ],
  "applies_to": {"platform": "apple|all", "build": "documented|shipped|both", "model_rev": ""},
  "formula": "for numbers: how it's computed from cited inputs",
  "conflicts_with": ["existing doc and what it says"] }
```

ID prefixes: `LM-ENG` engine and sessions, `LM-EXE` executors, `LM-CNV` conversation, `LM-CMP`
components, `LM-FMT` file format, `LM-CAPI`, `LM-SWIFT`, `LM-BLD` build, `LM-VER` versions, `RT-RUN`
LiteRT runtime, `RT-DLG` ML Drift delegate, `RT-CPU`, `MD-ARC` ML Drift architecture, `MD-KRN`
kernels, `MD-MTL` Metal backend, `LM-OTH`/`RT-OTH`/`MD-OTH` other platforms, `MOD-E4B`, `MOD-E2B`,
`MOD-FAM`, `MOD-EMB`, `MOD-TOK`, `APL` Apple, `OUR` patches, `PRC` process, `X` cross-cutting.
IDs are the prefix plus three digits (`LM-ENG-007`); `_work/groups.json` says which group owns which
prefixes and pages.

Files: a group's claims go to `_work/claims/<group>.jsonl` (one JSON object per line), verdicts to
`_work/verdicts/<group>.<batch>.jsonl`, and `tools/agello-docs/claims.py merge <group>` writes
`ledger/<group>.jsonl` (claim, verdict, verifier note; corrected claims keep `original_text`). Tier-B
claims outside the verification sample are `unsampled` in the ledger and carry no tag in the page.
`claims.py check <group>` validates the claims file and the tags; `claims.py show` prints claims with
their cited lines for verifiers. Code sources in other repos: `{"type": "agello", "path": "...",
"commit": "<Agello commit>", "lines": [a, b]}` for files in the Agello app repository.

**Tier A** (all verified): numbers; model architecture; anything that could end up in user docs,
commit messages or upstream text; anything that contradicts an existing doc. **Tier B**: other
structural and behavioral statements (30% sample verified, more if errors show up).

## Evidence rules

| To claim | You need |
| --- | --- |
| What our build does | Code at the pin (cited), the call path hop by hop; for "always/never/only", a search for counterexamples |
| That something happens on Apple | The code path and the condition that makes it active there (`#if __APPLE__`, `select()` on ios/macos, a Metal-only class, a runtime check) |
| A number | The inputs (cited) and the formula; recompute it |
| What a model file contains | `generated/models/*` (its bytes) |
| What the model is (trained) | `generated/hf/*/config.json` at its revision, or the model card / technical report |
| What Google says about performance | Attribute and date it: "Google reports ..., as of 2026-10-15" |
| A measurement of ours | A file in `docs/agello/results/` and its conditions. Don't run new performance measurements |

Precedence when sources disagree: for our build, code at the pin > an empirical run > anything else;
for a file, its bytes; for the model, the official config or card (keep trained and exported values
apart); older Agello notes (AUDIT, tracks, TELEMETRY, VALIDATION, PARITY, skills) are leads to
check, never sources by themselves. Measurement files in `docs/agello/results/` are the exception.
When two numbers measure different things, keep both and say what each measures.

## Writing

- Plain, specific English. Short paragraphs, tables for comparisons, numbered steps for procedures.
- Name things as the code does (`SessionAdvanced`, `odml.runtime_bmm`) and explain once.
- Distinguish: *trained context* (the model's), *export window* (the file's 32003 magic number,
  rewritten at load), *allocated context* (`maxNumTokens`), *filled length* (tokens in the KV cache).
- Say "GPU Neural Accelerators" for the matrix units ML Drift reaches through `ConvAppleMPP` on
  Apple10 GPUs; never "Neural Engine" (the ANE), which this stack doesn't use unless shown.
- Mark Agello-specific changes: "(ours: `<commit>`)".
- No marketing words, no "simply", no unexplained acronyms, no future promises.
- Dates absolute (2026-10-15). Units with a space (2.2 GB, 28,672 bytes, 41.2 tok/s).

## Out of bounds for writers

- Don't edit pages you don't own, existing docs outside `reference/`, code, or anything in the Agello repo.
- Don't post anything anywhere, run new performance measurements, download models, or build with Bazel.
- Delete any scratch you create; disk space is limited.
