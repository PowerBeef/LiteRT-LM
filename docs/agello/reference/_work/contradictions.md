# Contradictions dossier (for Run D)

> **Scope:** every contradiction between existing docs and the verified reference: C1-C17 from
> `_work/outline.md` and the new ones the Run A and B researchers reported. For each: the values seen
> and exactly where, the resolution, the claims and fact behind it, and the precise change to make in
> each existing doc. **Out of scope:** making the changes (Run D does, with the user's OK); the facts
> themselves ([../facts.md](../facts.md)).
> **Pinned to:** LiteRT-LM `7b872544`, LiteRT `362759838`, ML Drift `15fa9e2`; fork docs read on
> branch `agello` at `61c03d8e`; Agello read at `93122b6`.
> **Applies to:** the fork's `docs/agello/*` and Agello's docs. **Last verified:** 2026-10-15

## Summary

- 17 assigned contradictions (C1-C17) and 44 new ones (N1-N44). All C items are resolved; 3 new
  ones stay open (N22 float16 in EmbeddingGemma, N40 the `.bazelrc` comment, N41 the drafter key
  scales' effect) and are marked so.
- Most resolutions are "both values are right about different things": trained vs exported vs
  allocated context, an early vs a steady memory reading, Google's binary vs our build, a build before
  vs after a commit. The fix is usually to name the condition, not to change the number.
- **The user's OK comes first.** Run D needs the user's explicit OK before changing anything outside
  `docs/agello/reference/`: the fork's other docs (`docs/agello/*.md`, `tracks/`, `upstream/`,
  `tools/agello-bench/README.md`, `.github/README.md`) and, above all, every file in the Agello
  repository (`/Users/patricedery/Coding projects/iRT`: README, `CLAUDE.md`, `docs/`, `.claude/skills/`,
  code comments). Nothing here authorizes an edit by itself.
- Locations: fork paths are relative to the LiteRT-LM checkout (branch `agello`, `61c03d8e`), the
  landing page as `agello:.github/README.md`; Agello paths are prefixed "Agello" and read at `93122b6`.
  Line numbers are those commits'.

## Concepts

| Change type | Meaning |
| --- | --- |
| **edit** | Replace the stated text with the given wording (or the fact's safe wording from facts.md). |
| **banner** | Leave the historical text, add a one-line note at the top or beside it saying which build or date it describes and where the current value lives. Use for dated logs (AUDIT, tracks, VALIDATION entries, drafts) whose text records what was true then. |
| **upstream note** | Not our doc to edit; a candidate line for an upstream issue or a draft, posted only with the user's OK. |
| **none** | The existing doc is right; listed so nobody "fixes" it. |

## Index

| ID | Topic | Resolution in one line | Fact | Docs to change |
| --- | --- | --- | --- | --- |
| [C1](#c1-app-memory-limit-iphone-17-pro) | App memory limit | 8 GiB steady; 6 GiB is an early reading | F-071, F-072 | VALIDATION, results README, archived plan |
| [C2](#c2-vision-tokens-per-image) | Vision tokens | 280 max in the file; 560/1,120 are the trained model | F-064 to F-066 | USER_GUIDE, ChatTypes comment, archived plan |
| [C3](#c3-patches-in-the-shipped-wrapper) | Wrapper patches | two | F-003 | Agello CLAUDE.md |
| [C4](#c4-fork-landing-page) | Landing page | stale on four points | F-011, F-015 | `.github/README.md` |
| [C5](#c5-re-pinned-builds-byte-identical) | Re-pinned builds | not byte-identical | F-012 | agello-litert-update skill |
| [C6](#c6-agello-bench-readme) | agello-bench README | stale on three points | F-010, F-015 | `tools/agello-bench/README.md` |
| [C7](#c7-test-counts) | Test counts | Agello facts; README is older than the tests | F-130 | Agello README, TESTING |
| [C8](#c8-upstream-drafts) | Draft count | 4 issues + 4 PRs | F-126 | PLAN-MAC |
| [C9](#c9-e4b-kv-cache-at-32k) | KV at 32K | 918 MB at 32,000 positions; 940 MB is 32,768 | F-058 | E4B-GRAPH, T4 |
| [C10](#c10-cpu-sampling-cost) | Sampling cost | two builds of the same span | F-114 | TELEMETRY, T2 |
| [C11](#c11-kept-conversations-kv-copy) | Kept KV copy | Agello before/after `6d389e6` | F-079 | T3 |
| [C12](#c12-window-aware-kernels) | Window kernels | five kernels at `15fa9e2` | F-096 | T4 |
| [C13](#c13-fp16-logits) | FP16 logits | follows the sampler setting | F-097 | AUDIT |
| [C14](#c14-context-length) | Context length | 128K trained, 32K exported | F-048 to F-050 | archived plan, InferencePlanner comment |
| [C15](#c15-audit-line-numbers) | AUDIT line numbers | v0.18.0 (`agello`) | — | AUDIT |
| [C16](#c16-performance-figures) | Performance | README = Google's v0.18.0 | F-106 to F-109 | Agello README, ENGINEERING |
| [C17](#c17-model-revision-in-audit) | Model revision | `13d5aa4c` is the older 2048-block file | F-020, F-022 | AUDIT, BASELINE |
| [N1-N3](#from-the-apple-platform-research-b3) | Simulator, Mac `availableBytes`, envelope | see items | F-010, F-075, F-077 | CLAUDE.md, TESTING, README, DeviceProfile comment, ENGINEERING |
| [N4-N8](#from-our-changes-and-process-b4) | Skill on PRs, drafts | see items | F-126 to F-129 | skill, drafts |
| [N9-N13](#from-the-models-research-b1) | E4B/E2B export details | see items | F-039, F-046, F-049 | E4B-GRAPH, AUDIT, InferencePlanner comment |
| [N14-N17](#from-the-family-and-tokenizer-research-b2) | Template and KV table | see items | F-032, F-061 | (upstream notes) |
| [N18-N22](#from-the-embeddinggemma-research-b2) | EmbeddingGemma | see items | F-056, F-123 | (none in our docs) |
| [N23-N44](#from-the-code-research-run-a) | Code-level notes | see items | various | AUDIT, TELEMETRY, Agello comments, upstream notes |

## C1: App memory limit (iPhone 17 Pro)

- **Values seen.** 8.6 GB: Agello `docs/ENGINEERING.md:73` ("about 8.6 GB"), the harness device
  header of 30 of 36 reports. 6.4 GB: `docs/agello/VALIDATION.md:7` ("app limit 6.4 GB"),
  `docs/agello/results/iphone/2026-10-15/README.md:3`, Agello `docs/archive/agello-overhaul.md:44`
  ("~6.4 GB"). 3.5 GB unentitled: Agello `Agello/Agello.entitlements:10` comment.
- **Resolution.** Both come from the same reading (footprint + `os_proc_available_memory()`) with the
  increased-memory-limit entitlement. 8,589,934,592 bytes (8 GiB, printed "8.6 GB") is the limit for
  the life of the app; 6,442,450,944 bytes (6 GiB, "6.4 GB") is what the reading returned early in 6
  of 36 runs, before the first engine load. VALIDATION copied the first report's header. The
  unentitled limit and the archived plan's earlier "~6.4 GB" have no raw record.
  <!-- c:APL-001 --> <!-- c:APL-002 --> <!-- c:APL-003 --> <!-- c:APL-004 --> <!-- c:APL-005 --> <!-- c:APL-006 -->
- **Claims:** APL-001 to APL-006, APL-024. **Facts:** F-071, F-072, F-073.
- **Changes.**
  - `docs/agello/VALIDATION.md:7`: **edit** "app limit 6.4 GB" → "app limit 8.6 GB (8 GiB; the
    harness header read 6.4 GB, an early reading, see reference/facts.md F-071)".
  - `docs/agello/results/iphone/2026-10-15/README.md:3`: **edit** the same way (results files stay
    unchanged; only the README's summary line).
  - Agello `docs/archive/agello-overhaul.md:44`: **banner** "Superseded: the steady limit is about
    8.6 GB (8 GiB); ~6.4 GB was an early reading" (archived plan, don't rewrite).
  - Agello `Agello/Agello.entitlements:10`: **edit** the comment to "Default limit unmeasured" or
    remove the number (needs a measurement to restore it).
  - Agello `docs/ENGINEERING.md:73`: **none** for the 8.6 GB (see N3 for the same line's envelope
    claim).

## C2: Vision tokens per image

- **Values seen.** 70/140/280: AUDIT, the file's `vision_70/140/280` signatures. 140/280/560: Agello
  `docs/USER_GUIDE.md:41`. 70-1120: Agello `Agello/Inference/ChatTypes.swift:67` (comment).
  "up to 1,120": Agello `docs/archive/agello-overhaul.md:63`. 2,520 patches: file metadata.
- **Resolution.** The trained model accepts 70-1,120 per image; the E2B/E4B files export only 70,
  140 and 280 (2,520 patches = 280 × 9); LiteRT-LM (both builds) caps at 280 and rejects more with
  InvalidArgument; Agello clamps to `ModelInfo`'s 280, so "Detailed (560)" sends 280.
  <!-- c:MOD-E4B-019 --> <!-- c:MOD-E4B-046 --> <!-- c:MOD-E4B-049 --> <!-- c:MOD-E4B-050 --> <!-- c:MOD-E4B-051 -->
- **Claims:** MOD-E4B-019, -046, -049, -050, -051, -052. **Facts:** F-064, F-065, F-066.
- **Changes.**
  - Agello `docs/USER_GUIDE.md:41`: **edit** "image detail (140 / 280 / 560 visual tokens)" →
    "image detail (140 or 280 visual tokens; Detailed uses 280 with the current model)".
  - Agello `Agello/Inference/ChatTypes.swift:67` (comment) and the "Detailed (560)" label (`:85`):
    **edit** to say the model allows 70-1,120 but the phone file accepts at most 280 (a code change:
    the user decides whether to relabel or drop the preset).
  - Agello `docs/archive/agello-overhaul.md:63`: **banner** (trained-model capability, not the phone file).

## C3: Patches in the shipped wrapper

- **Values seen.** One: Agello `CLAUDE.md:24` ("v0.18.0 plus one tool-call patch"). Two: Agello
  `Vendor/README.md`, `NOTICE`, the `agello-litert-update` skill, README:223.
- **Resolution.** Two Swift patches in `swift/Conversation.swift` (failed tool calls answered with an
  error; a stream holds its conversation weakly), plus a comment-only commit. `CLAUDE.md` is stale
  since Agello vendored `85b1a776` on 2026-10-14.
  <!-- c:OUR-002 --> <!-- c:OUR-003 --> <!-- c:LM-SWIFT-046 -->
- **Claims:** OUR-002, OUR-003, LM-SWIFT-019, -023, -046, LM-VER-001. **Fact:** F-003, F-004.
- **Changes.** Agello `CLAUDE.md:24`: **edit** "v0.18.0 plus one tool-call patch" → "v0.18.0 plus two
  Swift patches (tool-call errors, a stream's weak reference to its conversation)". Others: **none**.

## C4: Fork landing page

- **Values seen.** `agello:.github/README.md`: lines 11-13 list branches `agello`, `main` and
  `research/3913` (as a branch), no `agello-next`; lines 15-21 list one patch; lines 26-27 say "the
  Metal GPU accelerator is a prebuilt binary even upstream". The outline's "open since 2026-10-08".
- **Resolution.** The `agello` branch carries two Swift patches; the fork also has `agello-next`
  (`7b872544`); `research/3913` is an annotated tag. LiteRT's base has the Metal accelerator's source
  and our `agello-next` links it statically from source. The date ML Drift's source went public isn't
  established (oldest forks 2026-10-09 UTC). <!-- c:OUR-025 --> <!-- c:OUR-026 --> <!-- c:OUR-027 -->
- **Claims:** OUR-025, OUR-026, OUR-027, OUR-031, LM-VER-029. **Facts:** F-011, F-015.
- **Changes.** `agello:.github/README.md`: **edit** the branch table (add `agello-next`: "our source
  build of LiteRT-LM, LiteRT and ML Drift; not what Agello ships"; move `research/3913` to a "Tags"
  line); list both patches (add "A stream holds its conversation weakly"); replace the "prebuilt
  binary even upstream" sentence with "Agello links Google's prebuilt v0.18.0 binary; `agello-next`
  builds the runtime, including LiteRT's Metal accelerator, from source". Follow
  [../patches.md](../patches.md#the-fork-landing-page-c4). Pushing the default branch needs the
  user's OK. Anywhere "2026-10-08" is given as ML Drift's open-source date (`docs/agello/AUDIT.md:12`, `:26`, `:159`):
  **edit** to "early October 2026 (forks from 2026-10-09)".

## C5: Re-pinned builds byte-identical

- **Values seen.** "should come out byte-identical": Agello `.claude/skills/agello-litert-update/SKILL.md:164`.
  "isn't byte-identical; compare replies": `docs/agello/BUILD.md:15-17`.
- **Resolution.** BUILD.md is right. The rebuild stamps a different LiteRT-LM field (`+<diff hash>` or
  the re-pin commit) and so a different build id into the binary. BUILD.md's own "15/15 identical"
  check has no results file. <!-- c:LM-BLD-029 --> <!-- c:LM-BLD-030 -->
- **Claims:** LM-BLD-029, LM-BLD-030. **Fact:** F-012.
- **Changes.** Agello skill `SKILL.md:164`: **edit** "the framework should come out byte-identical"
  → "the framework gets a new build id; compare replies (agello-bench `short`), not bytes".
  `docs/agello/BUILD.md:17`: **edit** to add where the 15/15 record is, or mark it "not recorded".

## C6: agello-bench README

- **Values seen.** `tools/agello-bench/README.md:6` ("on the `research/3913` branch"); `:57-58`
  ("until our own builds add a macOS Metal slice"); `:14-15` (suites list of ten).
- **Resolution.** Our framework has had a macOS arm64 slice with the Metal accelerator linked
  statically since `3b2327cf`; `research/3913` is a tag; the `agello` tip `61c03d8e` adds suites
  `helper-turn`, `helper-turn-check`, `helper-turn-cycles`, `thinking-mask`. The WebGPU statement
  about Google's Mac binary is right. <!-- c:LM-BLD-063 --> <!-- c:LM-BLD-064 --> <!-- c:LM-VER-029 -->
- **Claims:** LM-BLD-063, LM-BLD-064, LM-VER-029, RT-OTH-005. **Facts:** F-010, F-011, F-015.
- **Changes.** `tools/agello-bench/README.md`: **edit** line 6 "branch" → "tag"; lines 57-58 →
  "our `agello-next` framework has a macOS Metal slice (since `3b2327cf`)"; lines 14-15: add the four
  suites.

## C7: Test counts

- **Values seen.** 220 unit + 48 device + 23 UI: Agello `README.md:168`, `docs/TESTING.md:9-11`.
  227 unit passed (+1 skipped), ui 22/22: `docs/agello/VALIDATION.md:13`, `:21` (2026-10-15).
- **Resolution.** The README and TESTING are older than the tests: at `93122b6` the sources declare
  227 / 49 / 23 test methods. Rule: test counts are Agello facts, stated only in Agello's README and
  TESTING as harness results with date and commit; the reference doesn't carry them.
  <!-- c:PRC-017 -->
- **Claims:** PRC-017. **Fact:** F-130 (the rule).
- **Changes.** Agello `README.md:168` and `docs/TESTING.md:9-11`: **edit** to the latest harness
  counts with date and commit (e.g. "227 unit tests passed on 2026-10-15, `93122b6`"); re-run the
  suites first if the counts are to include `AgelloDeviceTests` (49 declared, no full count recorded).
  The README badge, if it carries a count, changes with them.

## C8: Upstream drafts

- **Values seen.** "Drafts for 3 LiteRT-LM issues and 3 LiteRT/ML Drift PRs": `docs/agello/PLAN-MAC.md:30`
  and the subject of `8a1df223`. `docs/agello/upstream/README.md`: four issues, three LiteRT PRs, one
  ML Drift PR.
- **Resolution.** Seven drafts: four LiteRT-LM issues, three LiteRT pull requests, one ML Drift pull
  request inside the bounded-attention draft; none posted. <!-- c:OUR-016 --> <!-- c:PRC-007 -->
- **Claims:** OUR-016, PRC-007. **Fact:** F-126.
- **Changes.** `docs/agello/PLAN-MAC.md:30`: **edit** → "Drafts for 4 LiteRT-LM issues, 3 LiteRT pull
  requests and 1 ML Drift pull request; none posted". The commit subject stays.

## C9: E4B KV cache at 32K

- **Values seen.** 918 MB: `docs/agello/AUDIT.md:45`, `:227`. 940 MB: `docs/agello/E4B-GRAPH.md:104`,
  `docs/agello/tracks/T4-attention.md:108`.
- **Resolution.** 28,672 bytes per position. A 32K request (`maxNumTokens` 32,768) allocates 32,000
  positions = 917,504,000 bytes (AUDIT's 918 MB, decimal); 940 MB is 32,768 × 28,672 = 939,524,096,
  which this file never gets. In MiB: 875 and 896. <!-- c:MOD-E4B-037 --> <!-- c:MOD-E4B-038 --> <!-- c:MOD-E4B-039 --> <!-- c:LM-EXE-003 -->
- **Claims:** MOD-E4B-037, -038, -039, LM-EXE-003, LM-EXE-006. **Fact:** F-058.
- **Changes.** `docs/agello/E4B-GRAPH.md:104`: **edit** "about 300 MB instead of 940 MB" → "about
  280 MB instead of 918 MB (32,000 allocated positions)". `docs/agello/tracks/T4-attention.md:108`:
  **edit** "from 940 to about 300 MB" → "from 918 to about 280 MB". AUDIT: **none**. (The 280 MB is
  the trained-model arithmetic at 32,768, F-061; "about 300" is acceptable if preferred.)

## C10: CPU sampling cost

- **Values seen.** 0.50 ms: `docs/agello/TELEMETRY.md:116`, `docs/agello/tracks/T2-host-costs.md:3`.
  0.044 ms: `docs/agello/tracks/T7-sampler.md:17`.
- **Resolution.** Both are means of the same `sample` span (greedy CPU `TopPSampler` including the
  logits read, M6, level 2, MTP off): 0.50 ms on a build before direct logits reads
  (`results/mac/2026-10-14/bench/tele-t2k.jsonl`), 0.044 ms after T5's direct reads and the integer
  argmax (`results/mac/2026-10-15/bench/t6-plain.jsonl`). Neither measures k > 1.
  <!-- c:LM-CMP-039 --> <!-- c:LM-CMP-040 --> <!-- c:LM-CMP-041 -->
- **Claims:** LM-CMP-039, -040, -041. **Fact:** F-114.
- **Changes.** `docs/agello/TELEMETRY.md:116`: **banner** beside the row "measured 2026-10-14 before
  direct logits reads; 0.044 ms since (T7)". `T2-host-costs.md:3`: **none** (a dated track log).

## C11: Kept conversation's KV copy

- **Values seen.** "the copy stays until the chat's next message": `docs/agello/tracks/T3-helpers.md:92`.
  "lives only while the helper runs": `docs/agello/VALIDATION.md:99`, `:237-238`; Agello `6d389e6`.
- **Resolution.** In our build the snapshot lives until the kept session next acquires the executor.
  T3 described Agello before `6d389e6` (next acquisition = the next message); since `6d389e6`
  Agello's `endHelper` calls `getTokenCount()`, which restores and frees it right after the helper.
  <!-- c:LM-ENG-064 --> <!-- c:LM-ENG-065 --> <!-- c:LM-ENG-066 -->
- **Claims:** LM-ENG-064, -065, -066 (and -056, -059, -060). **Facts:** F-079, F-078.
- **Changes.** `docs/agello/tracks/T3-helpers.md:92`: **banner** "Agello before `6d389e6`; since then
  the copy is freed right after the helper (VALIDATION 2026-10-15)". VALIDATION: **none**.

## C12: Window-aware kernels

- **Values seen.** FC, softmax, ConvAppleMPP "per K tile": `docs/agello/tracks/T4-attention.md:78-79`.
  Also ConvWaveMatrix: `docs/agello/VALIDATION.md:85`, `:125` (ML Drift `15fa9e2`).
- **Resolution.** At `15fa9e2` the window-aware kernels are `FullyConnected`, `Softmax`,
  `Softmax1x1`, `ConvAppleMPP` (only when it tiles K itself) and `ConvWaveMatrix`; T4 predates
  `15fa9e2`. On Apple10, E4B's prefill query-key products run `ConvAppleMPP` without manual K tiling,
  so only the value products and the reduction skip keys there. Kernels without the window stay
  correct because keys below it are masked.
  <!-- c:MD-KRN-061 --> <!-- c:RT-DLG-037 --> <!-- c:MD-KRN-059 --> <!-- c:MD-KRN-060 -->
- **Claims:** MD-KRN-056 to -061, RT-DLG-034, -035, -037. **Fact:** F-096.
- **Changes.** `docs/agello/tracks/T4-attention.md:78-79`: **banner** "written before ML Drift
  `15fa9e2`, which adds `ConvWaveMatrix`; on Apple10 the prefill QK products don't use the window
  (reference/ml-drift/kernels.md)". VALIDATION: **none**. (N44 covers T4's "Next" bullet.)

## C13: FP16 logits

- **Values seen.** "FP16 logits only with FP16 precision and the GPU sampler (executor 1982-1998)":
  `docs/agello/AUDIT.md:113` (v0.18.0). "Greedy over FP16 logits in the CPU sampler": T7, ours.
- **Resolution.** AUDIT's condition is right if "GPU sampler" means the configured sampler backend,
  which is GPU by default on a GPU engine; the logits are FP16 even when the closed GPU sampler
  doesn't load and the CPU `TopPSampler` reads them (v0.18.0 converts to float; ours picks greedy over
  the halves, `fdd5bac7`). <!-- c:LM-CMP-028 --> <!-- c:LM-CMP-029 --> <!-- c:LM-CMP-030 --> <!-- c:LM-EXE-020 -->
- **Claims:** LM-CMP-028 to -031, LM-EXE-020. **Fact:** F-097.
- **Changes.** `docs/agello/AUDIT.md:113`: **edit** "the GPU sampler" → "the sampler backend set to
  GPU (the default on a GPU engine, whether or not the GPU sampler loads)".

## C14: Context length

- **Values seen.** "128K context": Agello `docs/archive/agello-overhaul.md:47`. "Gemma 4 E2B/E4B
  support 128K": Agello `Agello/Inference/InferencePlanner.swift:186` (comment). 32K: the export and
  the litert-community cards. `max_position_embeddings` 131072: config.
- **Resolution.** Three values for three things: trained 131,072 ("128K"); export window 32003 ("up
  to 32k"); allocated `maxNumTokens` below 32,003, else 32,000 (4,096 if unset).
  <!-- c:MOD-E4B-056 --> <!-- c:MOD-FAM-013 --> <!-- c:MOD-FAM-042 --> <!-- c:MOD-E2B-034 -->
- **Claims:** MOD-E4B-056, -057, -073, MOD-E2B-034, MOD-FAM-013, -042. **Facts:** F-048, F-049, F-050.
- **Changes.** Agello `InferencePlanner.swift:186`: **edit** the comment → "Gemma 4 E2B/E4B are
  trained for 128K, but the phone files export a 32K window (32,000 allocated at most); the practical
  ceiling is device memory". Agello `docs/archive/agello-overhaul.md:47`: **banner** ("the trained
  model; the phone file is 32K").

## C15: AUDIT line numbers

- **Values seen.** `docs/agello/AUDIT.md:10` lists LiteRT-LM "v0.18.0 (branch `agello`; the version
  Agello ships); upstream `main` as of 2026-10-14"; `:17` "Line numbers below are from `git show
  agello:<path>` unless noted". The reference cites `agello-next` `7b872544` (and Google's `df6df5b8`).
- **Resolution.** AUDIT's LiteRT-LM code locations are line numbers in the `agello` branch (v0.18.0
  source, with Agello's Swift patches in `swift/Conversation.swift`), the source of what Agello
  ships, not of the documented build; many of its statements still hold, but several describe v0.18.0
  only (N24-N26, N28, N31, N36 below) and its line numbers don't match the reference's permalinks.
  <!-- c:X-CON-001 -->
- **Claims:** X-CON-001 (new). **Fact:** none (a property of a document).
- **Changes.** `docs/agello/AUDIT.md`, top: **banner** "Line numbers and code descriptions are
  LiteRT-LM v0.18.0 (`agello` branch), what Agello ships. For the documented build (`agello-next`
  `7b872544`) see docs/agello/reference/; statements that changed are listed in its
  `_work/contradictions.md` (C9, C13, C17, N11, N24-N26, N28, N31, N36)." Don't renumber AUDIT.

## C16: Performance figures

- **Values seen.** Agello `README.md:91` "43 / 38 / 34 / 27 tok/s" (decode at 4K/8K/16K/32K, MTP) and
  `README.md:46` "~40 tokens/s"; Agello `docs/ENGINEERING.md:46` "43 → 27 tok/s from 4K to 32K";
  `docs/agello/VALIDATION.md` cooled calibration (2026-10-15); Agello `README.md:102` "Google reports
  1,189 prefill and 25 decode".
- **Resolution.** The README table measures Google's v0.18.0 binary (Agello's `Package.swift` has only
  named it; the decode row last changed before our first source-build commit). VALIDATION's cooled
  table measures our build `2e54fbdb9ca9`, four commits before the pin, with Google's binary run the
  same night (37.2 / 38.5 / 30.8 / 20.3 vs ours 38.7-40.1 / 39.7-42.2 / 39.4-46.8 / 37.3-44.6). Google's
  card figure (1,189 / 25.1 tok/s, no MTP, 2,048 context) is correctly attributed.
  <!-- c:LM-VER-023 --> <!-- c:LM-VER-024 --> <!-- c:LM-VER-025 --> <!-- c:LM-VER-026 --> <!-- c:X-FCT-001 -->
- **Claims:** LM-VER-023 to -026, X-FCT-001, X-FCT-002. **Facts:** F-106 to F-109.
- **Changes.** Agello `README.md:86` (the line under "## Performance"): **edit** to name the build:
  "… with speculative decoding (MTP), Google's LiteRT-LM v0.18.0 binary, measured before 2026-10-14".
  `README.md:102`: **edit** "25 decode" → "25.1 decode tokens per second … (Google's model card, as
  of 2026-10-15)". `docs/ENGINEERING.md:46`: **edit** "(43 → 27 tok/s from 4K to 32K)" → "(43 → 27
  tok/s from 4K to 32K with Google's v0.18.0 binary)". Agello `Agello/Inference/DeviceProfile.swift:60-62` (comment "Calibrated on iPhone 17 Pro … 43 → 38 → 34 → 27 tok/s"): **edit** to add "with Google's v0.18.0 binary". When Agello ships our build, replace the table
  from a fresh calibration with its build id. VALIDATION: **none**.

## C17: Model revision in AUDIT

- **Values seen.** `docs/agello/AUDIT.md:14` and `docs/agello/BASELINE.md:5`: revision `13d5aa4c`
  (SHA-256 `4a691f03…`). `pins.json` and Agello's ModelCatalog: `a1693e22` (`6353a233…`).
- **Resolution.** `13d5aa4c` is the first, 2048-token-block file (3,659,988,992 bytes); `a1693e22`
  (3,660,726,272 bytes) replaced it at the same file name on 2026-10-15. The older file isn't deleted:
  it stays downloadable at `13d5aa4c`. AUDIT's and BASELINE's measurements describe the 2048-block
  file. <!-- c:OUR-028 --> <!-- c:OUR-029 -->
- **Claims:** OUR-028, OUR-029, MOD-E4B-024, -069. **Facts:** F-020, F-022.
- **Changes.** `docs/agello/AUDIT.md:14`, `docs/agello/BASELINE.md:5`: **banner** "Measured with the
  2048-token-block revision `13d5aa4c`; Agello now uses `a1693e22` (512-token blocks, same weights)".

## From the Apple platform research (B3)

### N1: "The simulator has no Metal path for LiteRT-LM"

- **Values seen.** Agello `CLAUDE.md:7`, `docs/TESTING.md:3`, `README.md:178`; the reference's Run A
  pages before correction. Google's v0.18.0 xcframework has an `ios-arm64-simulator` slice that links
  Metal and contains the static Metal accelerator.
- **Resolution.** True of our framework (no simulator slice) and Agello's rule; not a property of the
  simulator or of Google's binary, whose simulator slice is untested. <!-- c:APL-066 --> <!-- c:APL-069 -->
- **Claims:** APL-066, APL-069, LM-BLD-031. **Fact:** F-010.
- **Changes.** Agello `CLAUDE.md:7`: **edit** "it has no Metal path for LiteRT-LM" → "our runtime has no
  simulator build, and the failures that matter only show on hardware". Same for `docs/TESTING.md:3`
  and `README.md:178`.

### N2: `DeviceProfile` comment on the Mac

- **Values seen.** Agello `Agello/Inference/DeviceProfile.swift:50`: "`availableBytes` 0 means not
  reported (the Mac)". The iPhone app on a Mac mini M6 reported 6.31-16.92 GB available in every Mac
  record of 2026-10-14. <!-- c:APL-020 --> <!-- c:APL-019 -->
- **Resolution.** On the Mac the reading is non-zero (the limit equals physical memory), so the
  keep-across-helpers memory test runs there too.
- **Claims:** APL-020, APL-019. **Fact:** F-075.
- **Changes.** Agello `DeviceProfile.swift:50`: **edit** the comment → "`availableBytes` 0 means not
  reported; the iPhone app on a Mac reports its physical memory". (Code comment: user's OK.)

### N3: Envelope validity and "validated by simulating 3.5 GB"

- **Values seen.** Agello `docs/ENGINEERING.md:73`: "The 8 GB floor (iPhone 15 Pro) was validated by
  simulating 5.0, 4.0 and 3.5 GB of usable memory"; `docs/TESTING.md:13` describes the suite.
- **Resolution.** Three scenarios read the early 6 GiB limit and tested 7.147, 7.147 and 6.147 GB
  instead of 5.0, 5.0 and 4.0. Where the target held, every 5.0 and 4.0 GB scenario recorded all
  steps; at 3.5 GB only the two host-idle runs did, and seven end after the first text turn.
  <!-- c:APL-025 --> <!-- c:APL-026 --> <!-- c:APL-027 -->
- **Claims:** APL-025, APL-026, APL-027. **Fact:** F-077.
- **Changes.** Agello `docs/ENGINEERING.md:73`: **edit** the last sentence → "Simulated 5.0 and
  4.0 GB limits completed every step on an iPhone 17 Pro; at 3.5 GB most runs stopped after the first
  text turn (2026-10-15)". Agello test (not a doc): the envelope test should re-read the limit or check
  it is ≈ 8.59 GB before allocating ballast (user's call).

## From our changes and process (B4)

### N4: Outside pull requests to LiteRT-LM

- **Values seen.** Agello `.claude/skills/agello-litert-update/SKILL.md:59`: outside pull requests
  "have been closed, not merged". GitHub: #2688 (CONTRIBUTOR) merged 2026-07-19; 30 of 2,497 merged PRs
  not by copybara-service. <!-- c:PRC-003 --> <!-- c:PRC-001 -->
- **Resolution.** The policy (issues only, CONTRIBUTING.md) stands; the "never merged" history doesn't.
- **Claims:** PRC-001, PRC-003. **Fact:** F-127.
- **Changes.** Skill `SKILL.md:59`: **edit** → "outside pull requests are rarely merged (30 of 2,497 by
  2026-10-15); CONTRIBUTING.md asks for issues".

### N5: Direct-copies draft vs upstream `bd326784`

- **Values seen.** `docs/agello/upstream/litert-metal-direct-copies.md` (write rows, around line 12)
  presents in-place writes as new. Google's LiteRT `bd326784` (2026-10-14) writes converted data in
  place on unlock for shared Metal buffers. <!-- c:OUR-011 -->
- **Resolution.** Google overlapped the write half (still converting first); our direct copies skip
  the conversion. The draft's write rows need rework before posting.
- **Claims:** OUR-011. **Fact:** F-133.
- **Changes.** The draft: **banner** "Upstream `bd326784` (2026-10-14) now writes in place on unlock;
  rework the write rows to show what ours adds (no conversion)".

### N6: XNNPACK patch drop condition reached upstream

- **Values seen.** `cpu-xnnpack.md` / WORKSPACE comment: "drop when TensorFlow's pin passes
  `c72e2a4a`". Upstream LiteRT-LM main `c05e67a2` pins TensorFlow `5d071809` (XNNPACK `0a66ebb5`, 219
  commits past the fix). <!-- c:OUR-012 -->
- **Resolution.** The condition is met upstream; the patch goes at our next rebase. Not an error in any
  doc.
- **Claims:** OUR-012, RT-CPU-016. **Fact:** F-129.
- **Changes.** None now; at the rebase, drop the patch and the WORKSPACE comment (maintenance.md).

### N7: ML Drift PR draft misses `15fa9e2`

- **Values seen.** `docs/agello/upstream/litert-bounded-attention.md:62` lists ML Drift commits `6ae7c68`,
  `2184a7f`; the window in `ConvWaveMatrix` is `15fa9e2`. <!-- c:OUR-017 -->
- **Resolution.** The draft is incomplete.
- **Claims:** OUR-017. **Fact:** F-126.
- **Changes.** The draft line 62: **edit** to add `15fa9e2` (window in `ConvWaveMatrix`).

### N8: Draft numbers without raw records

- **Values seen.** `docs/agello/upstream/litert-lm-swift-stream-stall.md:25-26` ("367 s instead of
  966 s"); `docs/agello/upstream/litert-weight-cache-key.md:12`, `:26` and
  `docs/agello/tracks/T11-caches.md:6`, `:44-45` (2.2 / 4.4 / 6.6 GB). <!-- c:PRC-008 -->
- **Resolution.** No results file holds these. Probe 14's records hold cache folder sizes: 2.47-2.56 GB
  with the content key, about 2.47 GB per KV size with the per-size key (7.41 GB after three sizes).
- **Claims:** PRC-008, OUR-023. **Facts:** F-081, F-116.
- **Changes.** Both drafts: **edit** the numbers to the recorded ones (F-081; F-116's 600.2 s → 0.111 s
  for the stall) or mark them "not recorded". T11: **banner**.

## From the models research (B1)

### N9: MTP drafter head, report vs export

- **Values seen.** Gemma 4 report v2: E2B/E4B drafters use a clustered top-k head (d × 4096). The E4B
  file's drafter has a full-vocabulary INT4 [262144, 256] head. <!-- c:MOD-E4B-023 --> <!-- c:MOD-E4B-043 -->
- **Resolution.** Both stand: they describe different artifacts (trained drafter vs exported file).
- **Claims:** MOD-E4B-023, -043, MOD-FAM-032. **Fact:** F-046.
- **Changes.** None in our docs; any text about the drafter's head says which.

### N10: Drafter key scales (extends E4B-GRAPH)

- **Values seen.** `docs/agello/E4B-GRAPH.md:85-86`: only the local drafter blocks use a mismatched key
  scale (layer 4's 0.005961 vs cache 22's 0.005955). The global block also uses layer 5's 0.0010908608
  instead of cache 23's 0.0010912281. E2B's drafter has no mismatch. <!-- c:MOD-E4B-044 --> <!-- c:MOD-E2B-037 -->
- **Resolution.** E4B-GRAPH is incomplete; the effect on MTP acceptance is unmeasured (see N41).
- **Claims:** MOD-E4B-044, MOD-E2B-037. **Fact:** F-046.
- **Changes.** `E4B-GRAPH.md:85-86`: **edit** to add the global block's mismatch (0.03% low).

### N11: "Prefill runs layers 0-23"

- **Values seen.** `docs/agello/E4B-GRAPH.md:72`, `docs/agello/AUDIT.md:235`. The prefill signatures
  run attention for layers 0-22 and only the K/V projection (with its input norm) of layer 23 (23
  SOFTMAX, 210 FC, 188 RMS norms). <!-- c:MOD-E4B-029 --> <!-- c:MOD-E4B-007 -->
- **Resolution.** Imprecise, not wrong: layer 23 runs only as far as writing its cache.
- **Claims:** MOD-E4B-029, -007, MOD-E2B-028. **Fact:** F-035.
- **Changes.** Both lines: **edit** → "Prefill runs layers 0-22 and the key/value projection of layer 23".

### N12: "Global layers feature unified Keys and Values" (E2B card)

- **Values seen.** The google/gemma-4-E2B-it card (`3e22461f`). E2B's config has `attention_k_eq_v`
  false and its file keeps separate global K and V caches. <!-- c:MOD-E2B-011 --> <!-- c:MOD-FAM-008 -->
- **Resolution.** The card's statement applies to 12B/26B-A4B/31B, not E2B or E4B.
- **Claims:** MOD-E2B-011, MOD-FAM-008. **Fact:** F-040.
- **Changes.** None in our docs (Google's card; an upstream note candidate for the model card).

### N13: `InferencePlanner` "E2B/E4B support 128K"

Same resolution as C14 (MOD-E2B-034); the change is listed there.

## From the family and tokenizer research (B2)

### N14: What ends a tool-call output

- **Values seen.** Report Table 11 ends it with `<tool_call|><turn|>`; Google's prompt page and the
  Hugging Face template follow a tool call with `<|tool_response>`. Both 50 and 106 are stop tokens in
  the files. <!-- c:MOD-TOK-023 --> <!-- c:MOD-TOK-024 --> <!-- c:MOD-TOK-026 -->
- **Resolution.** LiteRT-LM stops on either; which one E4B emits is unobserved (open).
- **Claims:** MOD-TOK-023, -024, -026. **Fact:** F-044.
- **Changes.** None.

### N15: Thoughts in tool-calling turns

- **Values seen.** Google: keep thoughts within a tool-calling turn. `agello-next`'s default channel
  filter and the file template remove them. <!-- c:MOD-TOK-021 --> <!-- c:MOD-TOK-045 -->
- **Resolution.** A documented departure; quality effect unmeasured.
- **Claims:** MOD-TOK-021, -045. **Fact:** F-032.
- **Changes.** None now; a candidate for an Agello measurement.

### N16: KV cache table in the Gemma 4 report

- **Values seen.** Report Table 3: +0.05 (E2B) and +0.14 (E4B) "Gb" at 32k. Config arithmetic: 103.8
  and 278.9 MB, matching the exported layout's global part; 12B/26B/31B agree. Open question in the
  E4B returns. <!-- c:MOD-FAM-043 --> <!-- c:MOD-FAM-045 --> <!-- c:MOD-E4B-033 -->
- **Resolution.** Both kept; the reference uses the arithmetic. The report's E2B/E4B figures are about
  half (halving only the global term reproduces them).
- **Claims:** MOD-FAM-043, -045, -047, MOD-E4B-033. **Fact:** F-061.
- **Changes.** None in our docs.

### N17: Turn structure around tools in the file template

- **Values seen.** The published E2B/E4B file template closes the model turn after a tool call and
  reopens `<|turn>model\n` after a tool response; Google's Hugging Face template, its prompt page and
  LiteRT-LM's repository template keep one model turn. <!-- c:MOD-TOK-030 --> <!-- c:MOD-TOK-036 --> <!-- c:MOD-TOK-037 --> <!-- c:MOD-TOK-050 -->
- **Resolution.** The file's template is what runs (Agello included); recorded as a difference to
  test. Upstream report candidate after a quality measurement.
- **Claims:** MOD-TOK-030, -036, -037, -050, LM-CNV-027. **Fact:** F-032.
- **Changes.** `litert-lm/conversation.md` (reference, Run A) says the generation prompt is omitted
  after a tool response "so the model continues its turn": reword per the mod-family returns (a
  reference-internal fix for Run D or the gaps pass).

## From the EmbeddingGemma research (B2)

### N18: Sliding window 1024 vs 512

- **Values seen.** google/embeddinggemma-2 card: "Sliding Window 1024 tokens"; config:
  `sliding_window: 512`. <!-- c:MOD-EMB-016 --> <!-- c:MOD-EMB-017 -->
- **Resolution.** Same window: transformers treats 512 as a radius on each side. The export differs:
  signatures up to 1,024 have no window; 2,048 and 8,192 use −512..+511.
- **Claims:** MOD-EMB-016, -017, -043, -044, -045. **Fact:** —.
- **Changes.** None.

### N19: Context 8,192 vs 1,024

- **Values seen.** Card: 8,192-token context; config `max_position_embeddings` 262,144; LiteRT-LM loads
  up to 1,024 by default. <!-- c:MOD-EMB-004 --> <!-- c:MOD-EMB-013 --> <!-- c:MOD-EMB-050 -->
- **Resolution.** Three limits: trained 8,192, exported signatures to 8,192, loaded 128-1,024 by
  default (the file's `max_input_length`).
- **Claims:** MOD-EMB-004, -013, -039, -047, -050. **Fact:** F-056.
- **Changes.** None in docs; Agello's chunker limits characters, not tokens (lead for the app).

### N20: 130M + 140M split

- **Values seen.** Card: 130M transformer + 140M embedder. <!-- c:MOD-EMB-023 --> <!-- c:MOD-EMB-024 -->
- **Resolution.** Reproduced only if "embedder" includes the per-layer projection; our reading.
- **Claims:** MOD-EMB-023, -024. **Fact:** F-119.
- **Changes.** None.

### N21: Agello downloads EmbeddingGemma at `main`

- **Values seen.** Agello's catalog pins E4B to a commit but EmbeddingGemma to `main`. <!-- c:MOD-EMB-058 -->
- **Resolution.** Not a contradiction in a doc; USER_GUIDE says nothing about pinning. A lead: a moved
  `main` changes what users download.
- **Claims:** MOD-EMB-058. **Fact:** F-029.
- **Changes.** None in docs (a code decision for the user).

### N22: Float16 for EmbeddingGemma (open)

- **Values seen.** Card: "Do not use float16". LiteRT-LM runs the GPU encoder at FP16 by default.
  Agello's probe ranks correctly, scores within 0.001 across GPU paths. <!-- c:MOD-EMB-049 --> <!-- c:MOD-EMB-073 -->
- **Resolution.** **Open**: no FP32 comparison exists.
- **Claims:** MOD-EMB-049, -073, -068, -069. **Fact:** F-123.
- **Changes.** None until measured.

## From the code research (Run A)

### N23: Google's v0.18.0 "wipes" the chat's conversation

- **Values seen.** `docs/agello/tracks/T3-helpers.md:7`: "Google's v0.18.0 wipes the chat's
  conversation when the second one runs"; `docs/agello/PLAN-MAC.md:56`.
- **Resolution.** v0.18.0 snapshots the chat correctly but restores it by swapping the host snapshot
  in as the live state; the GPU delegate keeps its bound buffers, which hold the helper's cache.
  "Wiped" is the visible effect; the mechanism is a failed restore (fixed upstream by `76594366`).
  <!-- c:LM-ENG-055 -->
- **Claims:** LM-ENG-055. **Fact:** F-079.
- **Changes.** T3:7 and Agello `DeviceProfile.swift:42-43` (comment "Google's LiteRT-LM (v0.18.0) wipes the conversation"): **edit** "wipes" → "loses (its restore leaves the helper's cache in the GPU
  buffers)". PLAN-MAC:56: **none** (it describes Agello's own wipe-and-reseed).

### N24: Tokenizer built concurrently (AUDIT §2.2)

- **Values seen.** `docs/agello/AUDIT.md:81`. <!-- c:LM-ENG-005 -->
- **Resolution.** Only for files with an `llm_model_type` and with `parallel_file_section_loading`
  true (the default).
- **Claims:** LM-ENG-005. **Changes.** AUDIT:81: **edit** to add the two conditions.

### N25: "`tasks.cc:640` builds 262,144 strings" (AUDIT §7)

- **Values seen.** `docs/agello/AUDIT.md:271`. <!-- c:LM-ENG-082 -->
- **Resolution.** True of v0.18.0 and of upstream at the base; fixed by ours `5e946049`.
- **Claims:** LM-ENG-082. **Changes.** Covered by the C15 banner; optionally add "(fixed in
  `agello-next` `5e946049`)".

### N26: Decode-step history copy and ids buffer (AUDIT §2.3)

- **Values seen.** `docs/agello/AUDIT.md:102-104`. <!-- c:LM-EXE-013 --> <!-- c:LM-EXE-035 -->
- **Resolution.** Still what ships (v0.18.0); `agello-next` removed both (ours `e43f9a8d`).
- **Claims:** LM-EXE-013, -035. **Changes.** Covered by the C15 banner.

### N27: Passive GPU waits vs Agello's benchmark flag

- **Values seen.** `docs/agello/TELEMETRY.md:168-170` (2.0 ms instead of 29.3 ms of CPU per token with
  passive waits). Agello sets `enableBenchmark = true` (`ChatEngine.swift:142`), which requests active
  waits in both builds. <!-- c:LM-EXE-038 --> <!-- c:RT-RUN-028 --> <!-- c:OUR-034 -->
- **Resolution.** The commit (`645aaf65`) is right for engines without benchmark parameters; Agello's
  configuration spins.
- **Claims:** LM-EXE-038, RT-RUN-028, OUR-032, OUR-034. **Fact:** F-019.
- **Changes.** TELEMETRY:168-170: **edit** to add "applies only with benchmark mode off; Agello turns it
  on". Any user-facing or commit text about the saving says the same.

### N28: `Gemma4DataProcessor::Create` FailedPrecondition (AUDIT §3.4)

- **Values seen.** `docs/agello/AUDIT.md:182`. <!-- c:LM-CNV-073 --> <!-- c:LM-CNV-074 -->
- **Resolution.** True of v0.18.0 and upstream before our `d1488ce7`; at `agello-next` creation
  succeeds and only tool-call constraints are missing.
- **Claims:** LM-CNV-073, -074. **Changes.** AUDIT:182: **edit** to add "(v0.18.0; fixed in
  `agello-next` by `d1488ce7`)".

### N29: Agello comment "at the next message the runtime rewinds"

- **Values seen.** Agello `Agello/Inference/ChatEngine.swift:143-146`. <!-- c:LM-CNV-052 --> <!-- c:LM-CNV-054 --> <!-- c:LM-CNV-053 -->
- **Resolution.** True for the shipped v0.18.0 (lazy refill at the next message); with `agello-next` the
  refill happens right after each reply (upstream `97e6dc7d`).
- **Claims:** LM-CNV-052, -053, -054, -058. **Changes.** Agello comment: **edit** when Agello moves to
  our build ("after each reply" instead of "at the next message"); **none** before.

### N30: Docs quoting the Hugging Face template as what LiteRT-LM renders

- **Values seen.** Any doc that quotes `google/gemma-4-E4B-it`'s `chat_template.jinja` for these files
  (none found in Agello at `93122b6`). <!-- c:LM-CNV-027 -->
- **Resolution.** The files carry an older template; the file's is what runs.
- **Claims:** LM-CNV-027, MOD-TOK-030. **Fact:** F-032. **Changes.** None found; keep it so.

### N31: TopK Metal sampler condition (AUDIT §2.4)

- **Values seen.** `docs/agello/AUDIT.md:110-111`: used only when the environment carries a Metal
  device. <!-- c:LM-CMP-019 --> <!-- c:LM-CMP-020 -->
- **Resolution.** Correct, and also only when no WebGPU device option is set; tried by `dlopen`, then
  static pointers.
- **Claims:** LM-CMP-019, -020. **Changes.** AUDIT:110: **edit** (optional) to add the WebGPU condition.

### N32: `wait_until_done` "Added in version 0.2.0"

- **Values seen.** `c/conversation.h` at `agello-next` (Google's header). v0.18.0 (C API 0.2.0) neither
  declares nor exports it. <!-- c:LM-CAPI-038 --> <!-- c:LM-VER-007 -->
- **Resolution.** For Agello it arrives with C API 1.0.0. **Upstream note** candidate.
- **Claims:** LM-CAPI-038, LM-CAPI-036, LM-VER-007. **Fact:** F-007. **Changes.** None in our docs.

### N33: `convertCamelToSnakeCaseInToolDescription` "read only at creation"

- **Values seen.** `swift/ExperimentalFlags.swift` doc comment (Google's). `ToolManager.execute` reads it
  on every tool call. <!-- c:LM-SWIFT-028 -->
- **Resolution.** The code wins: set it once, before creating conversations. **Upstream note** candidate.
- **Claims:** LM-SWIFT-028. **Changes.** None in our docs.

### N34: XNNPACK fix date

- **Values seen.** Our notes and `b8f807ec`: 2026-09-23. The commit's author date is 2026-09-22 19:37
  −0700. <!-- c:RT-CPU-015 -->
- **Resolution.** Both right (UTC vs local). **Changes.** None.

### N35: "Upstream has the same code" (`51f099fb`)

- **Values seen.** Commit message of `51f099fb`. `b8f807ec` corrected it: upstream XNNPACK fixed it in
  `c72e2a4a`; TensorFlow's pin predated the fix. <!-- c:RT-CPU-012 --> <!-- c:RT-CPU-016 -->
- **Resolution.** Follow `b8f807ec`. **Changes.** None (commit messages stay).

### N36: LiteRT environment options (AUDIT §2.2)

- **Values seen.** `docs/agello/AUDIT.md:83`: "no library dir". <!-- c:RT-RUN-008 -->
- **Resolution.** Holds for CPU and GPU backends; NPU backends also get the dispatch and
  compiler-plugin directories. **Changes.** AUDIT:83: **edit** to add "(CPU and GPU backends)".

### N37: LiteRT cancellation API

- **Values seen.** LiteRT `compiled_model.h` declares `EnableCancellation()`/`Cancel()` without
  definitions; the C header points to functions that don't exist at `362759838`. <!-- c:RT-RUN-023 -->
- **Resolution.** Only the polling callback works. **Upstream note** candidate (LiteRT).
- **Changes.** None in our docs.

### N38: `gpu_accelerator` flag comment

- **Values seen.** LiteRT-LM `runtime/executor/BUILD` comment: "macOS uses WebGPU" by default. In our
  fork the static Metal accelerator is linked on macOS without consulting the flag. <!-- c:RT-RUN-005 --> <!-- c:RT-OTH-005 -->
- **Resolution.** The comment describes Google's setup (their v0.18.0 macOS binary is WebGPU).
- **Changes.** None (Google's file); our docs say which build.

### N39: Weight cache header version

- **Values seen.** LiteRT `serialization_weight_cache/GEMINI.md`: `kVersion` "incremented to 3";
  `cache_builder.h` defines 2. <!-- c:RT-DLG-054 -->
- **Resolution.** The code wins. **Upstream note** candidate. **Changes.** None in our docs.

### N40: `.bazelrc` comment on `build:macos` (open)

- **Values seen.** `build:agello`'s comment says the XCFramework's iOS slice doesn't use `build:macos`;
  `--enable_platform_specific_config` applies `build:macos` on any Mac host. <!-- c:LM-BLD-003 -->
- **Resolution.** **Open**: no build was run to check which flags reach the iOS slice.
- **Claims:** LM-BLD-003. **Changes.** None until checked.

### N41: Drafter key scales' effect on MTP (open)

See N10. Whether the mismatched scales lower acceptance is untested. **Open.**

### N42: Graph dumps and the telemetry level

- **Values seen.** `docs/agello/TELEMETRY.md:34` lists "graph dumps" under level 4; its own "GPU graphs"
  row and the code tie them only to `AGELLO_TELEMETRY_GRAPH_DIR`. <!-- c:MD-MTL-034 -->
- **Resolution.** Graph dumps are independent of the level.
- **Claims:** MD-MTL-034. **Changes.** TELEMETRY:34: **edit** remove "graph dumps" from the level-4 row
  (add "any level, with `AGELLO_TELEMETRY_GRAPH_DIR`").

### N43: "M6 reports Apple11" (AUDIT §8) vs ML Drift's Apple10

- **Values seen.** `docs/agello/AUDIT.md:15` ("Apple11 as Metal reports it") and §8; ML Drift labels
  the M6 Apple10. <!-- c:APL-033 --> <!-- c:APL-034 --> <!-- c:MD-ARC-003 -->
- **Resolution.** Not a contradiction: Apple's family vs ML Drift's. Pages say "ML Drift family".
- **Fact:** F-087. **Changes.** None (AUDIT already says "as Metal reports it").

### N44: T4 "Next": K/V conversion "over the whole capacity"; Run A brief "FP32 RoPE angles"

- **Values seen.** `docs/agello/tracks/T4-attention.md:106` (the INT8 K/V caches converted to FP16
  for prefill "over the whole capacity"); the Run A brief's "RoPE (FP32 angles from INT32)".
  <!-- c:MD-KRN-063 --> <!-- c:MD-KRN-075 --> <!-- c:MD-KRN-076 -->
- **Resolution.** The converter stops at the filled length (rounded up); what it doesn't skip is keys
  below the window. E4B's rotary angles are FP16 on the GPU at the default precision; ML Drift's FP32
  RoPE kernels exist but E4B doesn't use them.
- **Claims:** MD-KRN-063, -075, -076. **Fact:** F-098.
- **Changes.** T4:106: **edit** → "converted to FP16 for the prefill convolutions every run up to the
  filled length, including keys below the window". The brief: **none** (internal).

## Limits and failure modes

- Line numbers are the fork's `agello` branch at `61c03d8e` and Agello at `93122b6`; re-locate them
  by the quoted text if either moved.
- Lower-priority notes from the returns that change no existing doc (the `-gpu` error text, N-level
  details of the reference's own pages) are left to the gaps pass and to `_work/returns/*.md`.
- Leads in Agello's docs not checked by any claim (README "Time to first token ~0.4 s", "App memory
  footprint ~1.2-2.8 GB", ENGINEERING "0.7 GB less memory" and "10.25 GB" cache growth) are neither
  confirmed nor contradicted here.

## Open questions

- N22, N40, N41 above.
- C1: why the early 6 GiB reading happens; the unentitled limit.
- C7: the full device-scenario count from a harness run (49 declared at `93122b6`).
