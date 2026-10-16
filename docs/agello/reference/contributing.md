# Contributing upstream, and keeping our docs right

> **Scope:** how Google's three repositories take contributions (from their CONTRIBUTING.md and
> GOVERNANCE.md at our bases, and their history); the status of our drafts and of issue #3913;
> templates for issues, pull requests and commit messages; the evidence rules for anything we
> send upstream; the posting rule; license notices in our modified files; a checklist for
> Agello's user-facing docs, including where test counts belong (C7).
> **Out of scope:** what each patch does and when it can go, in [patches.md](patches.md); the
> #3913 model fix, in [models/gemma4-e4b.md](models/gemma4-e4b.md); building and releasing our
> framework, in [litert-lm/build-and-release.md](litert-lm/build-and-release.md).
> **Pinned to:** LiteRT-LM base `df6df5b8`, LiteRT base `ca21dc99`, ML Drift base `1b18771`
> (their contribution files); GitHub state read on 2026-10-15 (volatile).
> **Applies to:** all platforms. **Last verified:** 2026-10-15

## Summary

- **LiteRT-LM takes issues, not code.** Its CONTRIBUTING.md says the repository isn't ready for
  code contributions and asks for GitHub issues; that text is unchanged on `main` as of
  2026-10-15. Almost every merged pull request is Google's internal mirror (`copybara-service`).
- **LiteRT and ML Drift take pull requests** from people who have signed Google's CLA. LiteRT may
  apply a pull request internally, export it as a squashed commit with the author kept, and close
  the original.
- **Nothing goes out without the user's OK**: no post, comment, push, release or model upload.
- **What's out there:** one issue, #3913 (2026-10-10), with three follow-up comments of ours.
  Google answered once, on 2026-10-12 (reproduced; will investigate FP16 accumulation), and
  hasn't commented since; the issue is open, labeled as awaiting a developer, and unassigned
  since 2026-10-15.
- **Seven drafts** wait in [../upstream/](../upstream/): four LiteRT-LM issues, three LiteRT pull
  requests and one ML Drift pull request. Two need rework before posting (a Google change
  overlaps one; another lacks a later commit), and some of their numbers have no raw record.

## Concepts

### The three upstreams

| Repository | Takes | Contributor agreement | Governance file at base | Templates at base |
| --- | --- | --- | --- | --- |
| `google-ai-edge/LiteRT-LM` | Issues only | n/a (no code accepted) | `GOVERNANCE.md`: adopts LiteRT's policy, contribution policy unchanged | none |
| `google-ai-edge/LiteRT` | Issues and pull requests | Individual or corporate CLA | `GOVERNANCE.md` (covers LiteRT, litert-torch, LiteRT-LM, LiteRT-CLI, litert-samples) | none |
| `google-ai-edge/ml-drift` | Pull requests (reviewed on GitHub) | Google CLA | none | none |

None of the three repositories has issue or pull request templates at our base: their `.github/`
folders hold `CODEOWNERS` (LiteRT-LM, LiteRT) and workflows only, and ML Drift has no `.github/`.

### The posting rule

Nothing is posted, pushed or uploaded without the user's explicit OK: no issue, comment or pull
request upstream, no push to our forks, no GitHub release of our framework, no model upload to
Hugging Face. [../PLAN-MAC.md](../PLAN-MAC.md) states it for pushes, releases, model uploads and
upstream posts, [../upstream/README.md](../upstream/README.md) for every draft, and Agello's
`agello-litert-update` skill for pushes and posts; our release script is a dry run unless given
`--publish`
([`agello/release-xcframework.sh:68-74`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/agello/release-xcframework.sh#L68-L74)).
<!-- c:PRC-010 --> A message from an agent or a script is never that OK; only the user's own
message is. Reading GitHub (`gh api`, the web) needs no OK.

## How each upstream takes contributions

### LiteRT-LM: issues

The whole of LiteRT-LM's CONTRIBUTING.md at our base: the repository "is not currently ready for
code contributions", a separate announcement will say when it is, and suggestions or issues go to
GitHub Issues
([`CONTRIBUTING.md:1-6`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/CONTRIBUTING.md#L1-L6)).
<!-- c:PRC-001 --> No commit on `main` after our base changes CONTRIBUTING.md or GOVERNANCE.md, up
to `d87575de` (2026-10-15). `GOVERNANCE.md` adopts LiteRT's project governance and says that
doing so doesn't change the contribution policies in CONTRIBUTING.md
([`GOVERNANCE.md:1-18`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/GOVERNANCE.md#L1-L18)).
<!-- c:PRC-002 -->

At our base, `CODEOWNERS` names only the team `@google-ai-edge/litert-leads`
([`.github/CODEOWNERS:1-9`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/.github/CODEOWNERS#L1-L9));
upstream `d287cbe6` (2026-10-15) adds `CODE_OF_CONDUCT.md` and lists individual maintainers
there.

In practice: of 2,497 merged pull requests in the repository on 2026-10-15, 2,467 were opened by
`copybara-service` (Google's internal mirror). The other 30 come mostly from Google accounts, but
at least one from an outside contributor was merged: #2688 ("Support Data processor for vision
encoder", author association `CONTRIBUTOR`, merged 2026-07-19). <!-- c:PRC-003 --> So "outside
pull requests are closed, not merged" (Agello's `agello-litert-update` skill) is too strong; the
policy text still asks for issues, and an issue is what we file. Commits from the mirror carry a
`LiteRT-LM-PiperOrigin-RevId` trailer (for example `61dcfc77`).

What a LiteRT-LM issue of ours should carry: a reproduction with the public API (Swift or C), the
version and model revision, the device and OS, measured results with their conditions, and a link
to a commit in our fork that fixes it. Code goes no further than that link.

### LiteRT: pull requests under the CLA

LiteRT's CONTRIBUTING.md at our base asks for the individual or the corporate Contributor License
Agreement before any pull request; only original code from people who signed it is accepted.
Reviewers are assigned from the LiteRT team; approved pull requests that pass CI are merged, or
applied to Google's internal version control first and exported later as one squashed commit with
the pull request's creator as author, after which the pull request is closed and labeled pending
merge
([`CONTRIBUTING.md:5-25`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/CONTRIBUTING.md#L5-L25),
[`CONTRIBUTING.md:27-40`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/CONTRIBUTING.md#L27-L40)).
<!-- c:PRC-004 --> C++ follows the Google C++ Style Guide, checked with `clang-format
--style=google`, and new files start with a license header
([`CONTRIBUTING.md:42-46`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/CONTRIBUTING.md#L42-L46),
[`CONTRIBUTING.md:84-86`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/CONTRIBUTING.md#L84-L86)).

LiteRT's GOVERNANCE.md covers LiteRT, litert-torch, LiteRT-LM, LiteRT-CLI and litert-samples,
with Google keeping final authority through `@google-ai-edge/litert-leads`. Contributors may
submit issues, feedback and pull requests under the CLA and each repository's CONTRIBUTING.md
and CODE_OF_CONDUCT.md; maintainers decide on changes, and a disagreement can be escalated to
the project leads
([`GOVERNANCE.md:3-17`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/GOVERNANCE.md#L3-L17),
[`GOVERNANCE.md:19-36`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/GOVERNANCE.md#L19-L36),
[`GOVERNANCE.md:38-57`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/GOVERNANCE.md#L38-L57)).
<!-- c:PRC-005 --> No commit on LiteRT `main` after our base changes CONTRIBUTING.md,
GOVERNANCE.md or CODE_OF_CONDUCT.md, up to `309cf084` (2026-10-15).

### ML Drift: pull requests under the CLA

ML Drift's CONTRIBUTING.md at our base is Google's standard text: contributions need the Google
CLA (one signed for another Google project counts), the project follows Google's Open Source
Community Guidelines, and every submission is reviewed through a GitHub pull request
([`CONTRIBUTING.md:1-32`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/CONTRIBUTING.md#L1-L32)).
It has no GOVERNANCE.md, CODEOWNERS or code of conduct file at our base or on `main` (`6e803da`,
2026-10-15). <!-- c:PRC-006 --> Its first pull requests (#1 to #3, June 2026) are titled "Create
README.md for SD", "Test PR import config" and "Test w/ small lint change"; its oldest forks date
from 2026-10-09 (UTC).

### What signing the CLA involves

The CLA is signed once per person (or per employer, for the corporate one) at
`cla.developers.google.com`, and covers every Google project. It's the user's decision and
signature; nothing here signs it. Until it's signed, a pull request to LiteRT or ML Drift can't
be accepted, so posting one first only produces a failed CLA check.

## Our drafts

[../upstream/](../upstream/) holds seven drafts, added in one commit (`8a1df223`, 2026-10-15):

| Draft | Kind | Our commits | State on 2026-10-15 |
| --- | --- | --- | --- |
| [litert-lm-swift-stream-stall.md](../upstream/litert-lm-swift-stream-stall.md) | LiteRT-LM issue | `agello` `85b1a776` | Not posted |
| [litert-lm-swift-tool-call-errors.md](../upstream/litert-lm-swift-tool-call-errors.md) | LiteRT-LM issue | `agello` `a3c0d0ef` | Not posted |
| [litert-lm-session-snapshots.md](../upstream/litert-lm-session-snapshots.md) | LiteRT-LM issue | `aec81cb7`, `ed998d80`, `585203d3` | Not posted |
| [litert-lm-decode-host-costs.md](../upstream/litert-lm-decode-host-costs.md) | LiteRT-LM issue | `ef197a04`, `e43f9a8d`, `5e946049`, `fdd5bac7` | Not posted |
| [litert-bounded-attention.md](../upstream/litert-bounded-attention.md) | LiteRT PR and ML Drift PR | LiteRT `ede28c64`, `aa05bd2d`, `31386317`; ML Drift `6ae7c68`, `2184a7f` | Not posted; add ML Drift `15fa9e2` |
| [litert-weight-cache-key.md](../upstream/litert-weight-cache-key.md) | LiteRT PR | LiteRT `97e06d36` | Not posted |
| [litert-metal-direct-copies.md](../upstream/litert-metal-direct-copies.md) | LiteRT PR | LiteRT `e17213ea`, `584f3461` | Not posted; rework on top of LiteRT `bd326784` |

That is four LiteRT-LM issues and four pull requests (three LiteRT, one ML Drift), as the
patches page records (C8, OUR-016). As of 2026-10-15, the only issue or pull request by
`PowerBeef` in the three repositories is issue #3913. <!-- c:PRC-007 -->

Gaps to close before posting:

1. **Numbers without a raw record.** No record in [../results/](../results/) gives probe-suite
   durations or weight cache file sizes behind the drafts: the stream-stall draft's "367 s
   instead of 966 s" for the probe suite and the weight-cache draft's "2.2 → 4.4 → 6.6 GB" come
   from notes (the only suite times are the 2026-10-14 Mac README's inference runs, 139 s and
   119 s). The T11 bench records hold load times, footprints and cache counters, no cache sizes.
   The probe-14 records hold cache folder sizes: 2.47–2.56 GB for the content key (`kv-all`),
   and, with the per-size key (Mac `probes-t4`, `probes-p4`, `probes-t4b`), one folder of
   2.469–2.472 GB per KV size, about 2.47, 4.94 and 7.41 GB in total after the 8K, 4K and 16K
   loads. <!-- c:PRC-008 -->
   Either cite a results file or drop the number.
2. **Measured on working trees.** Several of the drafts' measurements come from builds with
   uncommitted changes ([patches.md](patches.md#evidence-and-its-limits), OUR-019); say so, or
   re-measure the commits.
3. **Upstream moved.** The direct-copies draft's write half overlaps Google's `bd326784`
   (OUR-011); the ML Drift draft lists `6ae7c68` and `2184a7f` but not `15fa9e2`, the window in
   ConvWaveMatrix (OUR-017).
4. **Mac versus iPhone.** The drafts' numbers are from the M6 Mac, and their index asks for an
   iPhone run before posting; the iPhone run of 2026-10-15 ([../VALIDATION.md](../VALIDATION.md))
   covers the T3, T11 and window builds, not each draft's table.
5. **Our names.** The drafts keep our environment switches (`AGELLO_BOUNDED_ATTENTION`,
   `AGELLO_SHARED_WEIGHT_CACHE`, ...) and track names; the weight-cache draft already says the
   name should change for upstream. Upstream text should propose options, not our variables.
6. **Links.** The index says links to our forks work "once the branches are pushed"; all three
   `agello-next` branches are pushed (LiteRT-LM `7b872544`, LiteRT `362759838`, ML Drift
   `15fa9e2`), so that caveat can go.

## Issue #3913: what Google has said

[google-ai-edge/LiteRT-LM#3913](https://github.com/google-ai-edge/LiteRT-LM/issues/3913)
("iOS GPU (default float16 activations): Gemma 4 E4B duplicates digits at ~3.7K-token context;
CPU and float32 are correct") was opened by `PowerBeef` on 2026-10-10. On 2026-10-15 it is open,
labeled `type :bug` and `status:awaiting LiteRT-LM developer`, with seven comments.
<!-- c:PRC-009 -->

| Date (UTC) | Who | What |
| --- | --- | --- |
| 2026-10-10 07:02 | `PowerBeef` | The report: digits duplicated past a few thousand tokens on the iOS GPU backend with v0.18.0; CPU and FP32 correct |
| 2026-10-12 00:17 | `john-rocky` (no association) | Reproduced on an M4 Max with v0.17.1 through the macOS WebGPU delegate; FP32 clears the failures |
| 2026-10-12 13:22 | `nireekshak` (`COLLABORATOR`) | Google's only reply (below) |
| 2026-10-13 23:27 | `PowerBeef` | The analysis: the rotary position cast from INT32 to float can't hold positions above 2,048 in FP16 |
| 2026-10-14 02:41 | `PowerBeef` | The model-side fix: positions split as `2048·h + l`, exact to 32,767 |
| 2026-10-14 03:44 | `PowerBeef` | A correction: the two WebGPU delegate problems don't reproduce as single ops |
| 2026-10-14 10:15 | `picitujeromanov` (no association) | An M1 Max loses FP16 precision below 2,048 too; 512-token blocks fix E2B and E4B |
| 2026-10-15 02:06 | `john-rocky` | Both patches exact on an M4 Max; no loss below 2,048 there |

<!-- c:PRC-011 -->

Google's one answer, from collaborator `nireekshak` on 2026-10-12: they reproduced the behavior
locally at longer context lengths and would investigate FLOAT16 accumulation precision in the GPU
attention and KV-cache kernels, and get back. <!-- c:PRC-012 --> The same day `nireekshak`
assigned the issue to themselves, then to `yichunk`, and added the two labels; `yichunk`
unassigned themselves on 2026-10-15 at 04:03 UTC, and no Google account has commented since
2026-10-12, so none has answered our analysis or fix. <!-- c:PRC-013 --> The skill's note that
Google hadn't replied to our comments (as of 2026-10-13) still holds on 2026-10-15.

Our analysis and the 512-token fix that followed the M1 Max report are what Agello ships (model
revision `a1693e22`; [patches.md](patches.md#model-revisions-in-older-notes-c17)). When writing
on the issue again (with the user's OK): answer the open question Google raised (accumulation in
attention and KV-cache kernels) with the evidence that the loss is upstream of them, in the
position tensor; link the fix's revision and its reproducer (`research/3913` tag); keep each
number tied to its device, runtime and model revision.

## Templates

Derived from our drafts and commits. Fill every bracket; drop a section that has nothing in it.

### Issue (LiteRT-LM)

```markdown
**Title:** <Component>: <what goes wrong, in the reader's terms>

**Body:**

<One paragraph: what an app does, what goes wrong, and why it matters.>

### Environment
- LiteRT-LM <version or commit>, <Swift | C> API
- Model: <repo> (`<file>`, revision `<rev>`, sha256 `<prefix>…`)
- <Device> (<chip>, <memory>), <OS and version>; backend `<gpu | cpu>`, `maxNumTokens` <n>

### Reproduction
<Steps or a minimal program; the call path with file:line permalinks at a released tag or `main`.>

### Results
| <Case> | <Before> | <After or expected> |
| --- | --- | --- |
<Conditions under the table: repetitions, warm-up, sampling, what was identical.>

### What we do
<The fix in two or three sentences, and its side effects.>
Patch: https://github.com/PowerBeef/<repo>/commit/<sha> (branch `<branch>`, <files>, <+/−>)
```

**Issue (LiteRT).** LiteRT takes issues as well as pull requests and has no issue template at our
base ([The three upstreams](#the-three-upstreams)). Use the same structure as the LiteRT-LM issue,
with two changes: the Environment line names the LiteRT commit (and the LiteRT-LM and ML Drift
commits built with it), and the Reproduction permalinks point to `google-ai-edge/LiteRT` at our base
`ca21dc99` or at `main`. A fix offered as a pull request falls under LiteRT's CLA
([F-128](facts.md#f-128)). <!-- c:PRC-004 -->

### Pull request (LiteRT, ML Drift)

```markdown
**Title:** <Area>: <the change, imperative or descriptive>

**Body:**

<Problem: what the code does today, with permalinks, and what it costs (measured).>

<Change: numbered steps of what the patch does; what it leaves alone; the switch that turns it
off, named as an option for upstream.>

### Results (<model, precision, device, OS>, <method: ABAB, warm-up + repetitions>)
| | Before | After |
| --- | --- | --- |

Fidelity: <exact-copy suites and counts; replies identical or not>.
Tests: <test target> (<n> cases).
Commits on `agello-next` of https://github.com/PowerBeef/<repo>: `<sha>`, ...
```

Before opening one: the CLA is signed, the branch is rebased on upstream `main`, our telemetry
calls and `AGELLO_*` switches are removed or turned into options, and new files carry Google's
license header.

### Commit messages (our forks and Agello)

Our commits follow one form: a subject `Area: what changed [tag]` (the plan or track tag in
brackets, for example `Build: LiteRT's Metal accelerator, linked statically, from source
[M1.2]`), a body that says what was wrong, what changed and what was measured with its machine
and conditions, then trailers: `Track:` and `Validated: <machine> <date>` on the fork commits that
were validated, and `Co-Authored-By:` and `Claude-Session:` lines. The author is the GitHub
noreply identity `109446870+PowerBeef@users.noreply.github.com`. <!-- c:PRC-014 -->

```text
<Area>: <what changed, one line> [<tag>]

<What was wrong, with the mechanism.> <What the change does.> <What it leaves alone.>

<Machine>, <model>, <window>, <MTP on/off>, <method>: <metric> <before> -> <after>
(<CI or spread>); replies identical. <test targets> pass.

Track: <tag>
Validated: <mac|iphone> <YYYY-MM-DD>

Co-Authored-By: ...
Claude-Session: ...
```

Agello's own commits use the same subject form without the tag (`Model: …`, `Harness: …`,
`Docs: …`), with the issue number when there is one (`(#3913)`). A commit subject that counts
things (drafts, patches, tests) is a fact like any other: count from the files (`8a1df223`'s
subject says "three LiteRT-LM issues" for four).

## Evidence rules for upstream text

Upstream text is read by people who can't see our notes. It follows [the style rules](_work/style.md)
and these:

1. **Every number has a raw record** in `docs/agello/results/` or a verified ledger claim, with
   its conditions: device and OS, runtime build id and commits, model file and revision, window
   (allocated context), prompt and output lengths, MTP on or off, sampling, warm-up and
   repetitions, and whether the arms ran in one binary (A/B) or as separate runs.
2. **Say which build.** Google's prebuilt v0.18.0, Google's nightly, or our source build at a
   commit; never mix them in one row without saying so ([versions.md](versions.md)).
3. **Code citations at upstream's commit.** `cite.py` links files we don't change to Google's
   repository at the base, so they paste into an issue; link our changes as commits in our fork.
4. **No claim beyond the evidence.** "Within noise" when the difference is inside the spread;
   "on the M6" when nothing else was measured; "always", "never", "only" after a search for
   counterexamples.
5. **Attribute others' results.** Name the commenter and their setup (as for the M1 Max finding
   on #3913); don't restate it as ours.
6. **One concern per issue or pull request**, as our drafts are split.
7. **Our private vocabulary out**: track names (T4, P3), build ids and `AGELLO_*` names explained
   or replaced.

## License notices in modified files

Apache 2.0 asks that modified files carry prominent notices that they were changed. Across the
three forks, our commits modify 57 existing C++, Objective-C++ and Swift files (28 in LiteRT-LM,
14 in LiteRT, 15 in ML Drift); 49 of them contain the word "Agello" (in comments next to the
changes, or in the telemetry includes), and one,
`swift/Conversation.swift`, has a header notice describing the change
([`swift/Conversation.swift:15-20`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L15-L20)).
The eight without a mention: LiteRT-LM `logit_mask.cc` and `tensor_buffer_util.cc`; LiteRT
`gpu_backend_metal_litert.mm` and `serialization_weight_cache.cc`; ML Drift
`fully_connected_util.cc`, `gpu_operation.cc`, `metal/common.mm` and `metal/compute_task.mm`.
<!-- c:PRC-015 --> Before we distribute a framework built from these sources, each modified file
should get a header notice like the Swift one; in an upstream pull request the notices come out
(the change becomes Google's file under the CLA).

## Checklist for user-facing docs

Agello's README, `docs/` (USER_GUIDE, ENGINEERING, ARCHITECTURE, TESTING), `NOTICE`,
`Vendor/README.md`, `CLAUDE.md` and the skills repeat facts that live here. On each change to
the runtime, the model or the tests, check:

| Fact | Where in Agello (`93122b6`) | Source of truth |
| --- | --- | --- |
| LiteRT-LM version | README badge (line 14), Project layout, "Vendored LiteRT-LM"; ARCHITECTURE code map ("prebuilt runtime") | [versions.md](versions.md) |
| Patches in the wrapper | README "Vendored LiteRT-LM" (two), `Vendor/README.md` (two), `NOTICE` (both), `CLAUDE.md` line 24 (one: stale, C3) | [patches.md](patches.md#what-agello-ships-c3) |
| Which runtime ships | ENGINEERING's #3913 section ("LiteRT-LM's runtime is a prebuilt binary") | [versions.md](versions.md#what-ships-and-what-is-documented) |
| Performance figures | README Performance table; ENGINEERING's #3913 results | [versions.md](versions.md#performance-figures-which-build) (C16) |
| Model file and revision | ModelCatalog, NOTICE, USER_GUIDE "The models", ENGINEERING | `pins.json`; [patches.md](patches.md#model-revisions-in-older-notes-c17) |
| Third-party licenses | `NOTICE` | When the app ships our framework: `agello/THIRD_PARTY_NOTICES.md` (LM-BLD-044) |
| Privacy manifest | `Agello/PrivacyInfo.xcprivacy` (`C617.1`, `35F9.1`) | [litert-lm/build-and-release.md](litert-lm/build-and-release.md#privacy-manifest) |
| Test counts | README badge and highlight, TESTING table | The harness, at a date (C7, below) |

At `93122b6`, Agello's README badge says LiteRT-LM 0.18.0, its "Vendored LiteRT-LM" section
lists two Swift patches, ARCHITECTURE calls the vendored package the "prebuilt runtime", and
ENGINEERING says LiteRT-LM's runtime is a prebuilt binary; all four match what Agello ships and
change when it ships our build. <!-- c:PRC-016 -->

Rules for those docs:

- **Name the build and date** with any speed or memory figure, as the README's Performance table
  needs (C16).
- **No reference-only facts in user docs** unless a verified claim backs them; link this
  reference from developer docs (`CLAUDE.md`, skills), not from the user guide.
- **Update together**: a wrapper change touches `Vendor/README.md`, `NOTICE`, the README section
  and `CLAUDE.md` in one commit (C3 came from missing `CLAUDE.md`).
- **The fork's landing page** (`.github/README.md` on `agello`) is a user-facing doc too (C4,
  [patches.md](patches.md#the-fork-landing-page-c4)).

### Test counts (C7)

Agello's README (badge and highlight) and TESTING.md give 220 unit tests, 48 device scenarios and
23 UI flows (291 in all). VALIDATION.md's iPhone run of 2026-10-15 reports `unit` 227 passed and
1 skipped, and `ui` 22 of 22 with the opt-in screenshot tour skipped. At Agello `93122b6` the test
sources declare 227 `func test…` methods in AgelloTests, 49 in AgelloDeviceTests and 23 in
AgelloUITests (22 in ChatFlowUITests plus the screenshot tour). <!-- c:PRC-017 -->

**C7 resolution (the rule):** test counts are Agello facts, not reference facts. They belong in
Agello's README and TESTING.md only, each stated as a measurement: the count a harness run
reported, with its date and Agello commit (for example "227 unit tests passed on 2026-10-15,
`93122b6`"). This reference doesn't carry them, and a run that skips opt-in tests (the screenshot
tour, lab-model probes) says so. The README's 220 / 48 are older than the tests added since.

## Limits and failure modes

- **GitHub state is volatile.** Issue comments, labels, assignees and pull request counts were
  read on 2026-10-15; re-read before quoting them.
- **Policies move.** Re-read each CONTRIBUTING.md on upstream `main` before posting anything.
- **A failed CLA check** blocks a pull request until the CLA is signed; posting first gains
  nothing.
- **Drafts age.** A draft that names commits and numbers goes stale as upstream moves (as the
  direct-copies draft did within a day).

## Where in the code

| Concept | File | Citation |
| --- | --- | --- |
| LiteRT-LM contribution policy | `CONTRIBUTING.md` | [`CONTRIBUTING.md:1-6`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/CONTRIBUTING.md#L1-L6) |
| LiteRT-LM governance | `GOVERNANCE.md`, `.github/CODEOWNERS` | [`GOVERNANCE.md:1-18`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/GOVERNANCE.md#L1-L18), [`.github/CODEOWNERS:1-9`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/.github/CODEOWNERS#L1-L9) |
| LiteRT CLA and pull requests | `CONTRIBUTING.md` | [`CONTRIBUTING.md:5-25`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/CONTRIBUTING.md#L5-L25), [`CONTRIBUTING.md:27-40`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/CONTRIBUTING.md#L27-L40) |
| LiteRT governance | `GOVERNANCE.md` | [`GOVERNANCE.md:19-36`](https://github.com/google-ai-edge/LiteRT/blob/ca21dc99de51c76bdcc572afeea1dd937d3c9e5c/GOVERNANCE.md#L19-L36) |
| ML Drift contribution policy | `CONTRIBUTING.md` | [`CONTRIBUTING.md:1-32`](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/CONTRIBUTING.md#L1-L32) |
| Release dry run | `agello/release-xcframework.sh` | [`agello/release-xcframework.sh:68-74`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/agello/release-xcframework.sh#L68-L74) |
| Our notice | `swift/Conversation.swift` header | [`swift/Conversation.swift:15-20`](https://github.com/PowerBeef/agello-litert-lm/blob/7b872544fdbd81308f98b1e2acf8c5798a549df6/swift/Conversation.swift#L15-L20) |

## Open questions

- **CLA and co-author trailers.** Our commits carry a `Co-Authored-By:` line naming a model, not a
  person. Whether Google's CLA check accepts commits with such a trailer, or the trailer has to be
  dropped from a pull request's commits, is not checked.
- **ML Drift's public date** (2026-10-08 or 2026-10-09, see
  [patches.md](patches.md#open-questions)) is not established from GitHub's records.
- **Outside contributions to LiteRT-LM.** At least one outside pull request was merged (#2688; 30 of 2,497 merged PRs weren't authored by copybara-service, PRC-003); whether
  that reflects a policy change or an exception isn't stated anywhere we found.
- **Whether a draft should be an issue or a pull request for LiteRT.** The snapshot and host-cost
  drafts are LiteRT-LM issues because LiteRT-LM takes no code; the code they describe can only be
  linked.
