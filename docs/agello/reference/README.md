# The Agello reference: LiteRT-LM, LiteRT, ML Drift and the models on Apple

> **Scope:** what this reference documents and at which commits, who should read what, a map of
> every page, the conventions the pages share, and how to regenerate and re-verify them.
> **Out of scope:** the facts themselves (the pages below; numbers in [facts.md](facts.md)); how
> Agello's app is built and tested (Agello's own `docs/` and skills).
> **Pinned to:** LiteRT-LM `7b872544`, LiteRT `362759838`, ML Drift `15fa9e2`; shipped: LiteRT-LM
> `85b1a776` with Google's prebuilt v0.18.0 binary.
> **Applies to:** Apple (iOS, macOS), with other platforms mapped in
> [other-platforms.md](other-platforms.md). **Last verified:** 2026-10-15

## Summary

- This reference explains how Agello runs Gemma 4 E4B and EmbeddingGemma 2 on Apple devices, from
  the Swift wrapper down to ML Drift's Metal kernels, with every statement tied to code at a pinned
  commit, a model file's bytes, an official model config or card, or a measurement file.
- It documents two things, and each page says which one a statement is about: the **documented
  build** (our source build on `agello-next`) and **what Agello ships** (the shipped wrapper with
  Google's v0.18.0 binary). They differ; [versions.md](versions.md#what-ships-and-what-is-documented)
  lists how.
- Start with [stack.md](stack.md) for the whole picture, [facts.md](facts.md) for numbers you
  want to quote, and [glossary.md](glossary.md) for the words to use.

## What is documented

The pins come from `tools/agello-docs/pins.json`, the single source of truth; a re-pin changes that
file first ([maintenance.md](maintenance.md#after-a-re-pin-or-rebase)).

| Part | Documented build | Base |
| --- | --- | --- |
| LiteRT-LM | `PowerBeef/agello-litert-lm` `agello-next` at `7b872544` | Google's `0.19.0-nightly.20261013` (`df6df5b8`) plus 29 commits of ours |
| LiteRT | `PowerBeef/LiteRT` `agello-next` at `362759838` | Google's `ca21dc99` (main, 2026-10-09) plus 12 commits |
| ML Drift | `PowerBeef/ml-drift` `agello-next` at `15fa9e2` | Google's `1b18771` (main, 2026-10-07) plus 8 commits |

| What Agello ships today | Exactly |
| --- | --- |
| Swift wrapper | `PowerBeef/agello-litert-lm` `agello` at `85b1a776`: v0.18.0 plus two Swift patches, vendored into Agello as `Vendor/LiteRT-LM` |
| Native binary | Google's prebuilt v0.18.0 `CLiteRTLM.xcframework` (closed build; LiteRT and ML Drift commits not recorded) |

| Model | File and revision |
| --- | --- |
| Gemma 4 E4B (what Agello downloads) | `gemma-4-E4B-it-fp16-rope.litertlm`, Google's file with 512-token rotary blocks, `PowerBeef02/gemma-4-E4B-it-litert-lm-fp16-rope` @ `a1693e22` |
| Gemma 4 E4B (Google's) | `litert-community/gemma-4-E4B-it-litert-lm` @ `2eee7ac3` |
| Gemma 4 E2B (headers and graphs only) | `litert-community/gemma-4-E2B-it-litert-lm` @ `b3ca0d2f` |
| EmbeddingGemma 2 text 270m | `litert-community/embeddinggemma-2-text-270m-litert-lm` @ `9be6e8b9` |
| Official configs | the `google/gemma-4-*` and `google/embeddinggemma-2` revisions in `pins.json` "hf_repos" ([generated/hf-configs.md](generated/hf-configs.md)) |

Our source build's runtime patches don't reach users until Agello ships that build; until then,
anything that differs between the two is marked with the build it applies to.

## Reading paths

**An Agello app developer** (changing `ChatEngine.swift`, `AppState` generation, budgets or
settings):

1. [stack.md](stack.md): one chat turn traced from Swift to Metal, threads, and who owns what.
2. [litert-lm/swift-api.md](litert-lm/swift-api.md) and [configuration.md](configuration.md): the
   API Agello calls and every setting, with what Agello sets.
3. [litert-lm/conversation.md](litert-lm/conversation.md) and
   [litert-lm/engine-and-sessions.md](litert-lm/engine-and-sessions.md): templates, thinking, tools,
   sessions, cancellation and KV snapshots.
4. [memory.md](memory.md) and [apple-platform.md](apple-platform.md): where the bytes go and the app
   memory limit.
5. [troubleshooting.md](troubleshooting.md) when something fails.

**A fork developer** (changing LiteRT-LM, LiteRT or ML Drift):

1. [versions.md](versions.md) and [patches.md](patches.md): what the pins are and every commit of ours.
2. The layer you change: [litert-lm/executors.md](litert-lm/executors.md),
   [litert/runtime.md](litert/runtime.md), [litert/ml-drift-delegate.md](litert/ml-drift-delegate.md),
   [ml-drift/architecture.md](ml-drift/architecture.md), [ml-drift/kernels.md](ml-drift/kernels.md),
   [ml-drift/metal-backend.md](ml-drift/metal-backend.md).
3. [litert-lm/build-and-release.md](litert-lm/build-and-release.md) and [tools.md](tools.md): building
   the XCFramework, agello-bench, telemetry and the GPU simulation switch.
4. [maintenance.md](maintenance.md): what to re-verify after your change moves cited code.

**Someone writing an upstream report** (an issue for LiteRT-LM, a pull request for LiteRT or ML Drift):

1. [contributing.md](contributing.md): how each upstream takes contributions, the posting rule, our
   drafts, templates and the evidence rules for upstream text.
2. [facts.md](facts.md) for every number you quote, with its conditions.
3. [versions.md](versions.md) to name the exact build, and [other-platforms.md](other-platforms.md)
   to read shared code and reports from other platforms.

**Someone writing user docs or commit messages:**

1. [facts.md](facts.md): each fact with a sentence safe to quote and the older figures it replaces.
2. [glossary.md](glossary.md): the wording to prefer and the wording to avoid.
3. [contributing.md](contributing.md#checklist-for-user-facing-docs) and its commit-message template.

## Page map

| Page | Answers |
| --- | --- |
| [stack.md](stack.md) | How a chat turn flows from Agello to Metal, on which thread, and how the documented build differs from what ships |
| [versions.md](versions.md) | What ships, what is documented, how upstream versions and tags, and what changed since v0.18.0 |
| [patches.md](patches.md) | Every commit of ours on the three forks and the shipped wrapper, its evidence and upstream status |
| [configuration.md](configuration.md) | Every setting in every layer, its default, its effect on Apple and what Agello sets |
| [memory.md](memory.md) | Where every byte of E4B and E2B lives at 4K to 32K allocated context |
| [apple-platform.md](apple-platform.md) | Apple GPU families, the app memory limit, Metal storage and argument buffers, the iPhone app on a Mac |
| [troubleshooting.md](troubleshooting.md) | Symptom, cause and fix for each failure mode, by build |
| [tools.md](tools.md) | The tools for inspecting files, running the engine, measuring builds, telemetry and this reference's scripts |
| [other-platforms.md](other-platforms.md) | What the three repositories do off Apple's Metal path |
| [facts.md](facts.md) | The canonical numbers and user-facing facts, with conditions and quotable sentences |
| [glossary.md](glossary.md) | The terms used here, what to prefer and what to avoid |
| [contributing.md](contributing.md) | How to contribute upstream, our drafts, templates, and checks for user-facing docs |
| [maintenance.md](maintenance.md) | How the reference stays true: the ledger, adding claims and pages, re-pins, corrections |
| [litert-lm/engine-and-sessions.md](litert-lm/engine-and-sessions.md) | Engine creation, sessions, threads, task lifecycle, cancellation, the resource manager and KV snapshots |
| [litert-lm/executors.md](litert-lm/executors.md) | The compiled-model executors: signatures, prefill chunking, allocated context, KV state, sampling hand-off, MTP |
| [litert-lm/conversation.md](litert-lm/conversation.md) | The Conversation layer: templates, Gemma 4 processing, channels and thinking, tools, constraints, history |
| [litert-lm/components.md](litert-lm/components.md) | Tokenizers, samplers, logit masks, constraints and embedding lookups |
| [litert-lm/file-format.md](litert-lm/file-format.md) | The `.litertlm` container, its sections and metadata, and external weights |
| [litert-lm/c-api.md](litert-lm/c-api.md) | The C API's conventions, objects, error reporting, versions and the changes from 0.2 to 1.0.0 |
| [litert-lm/swift-api.md](litert-lm/swift-api.md) | The Swift package: `Engine`, `Conversation`, streaming, tools, `ExperimentalFlags`, and how Agello uses it |
| [litert-lm/build-and-release.md](litert-lm/build-and-release.md) | Bazel, `--config=agello`, pins, our XCFramework, build stamping and releases |
| [litert/runtime.md](litert/runtime.md) | LiteRT's environment, compiled models, tensor buffers, events and accelerator registry |
| [litert/ml-drift-delegate.md](litert/ml-drift-delegate.md) | The ML Drift delegate in LiteRT: partitioning, composites, bounded attention, caches, the Metal glue |
| [litert/cpu-xnnpack.md](litert/cpu-xnnpack.md) | The CPU path: XNNPACK, KleidiAI, the XNNPACK weight cache and our XNNPACK patch |
| [ml-drift/architecture.md](ml-drift/architecture.md) | ML Drift's graph, compiled model, GPU detection and Apple families, precision and selectors |
| [ml-drift/kernels.md](ml-drift/kernels.md) | The kernels Gemma 4's GPU graphs use on each Apple family |
| [ml-drift/metal-backend.md](ml-drift/metal-backend.md) | How ML Drift encodes and runs work on Metal |
| [models/gemma4-family.md](models/gemma4-family.md) | The Gemma 4 models Google publishes and the design they share |
| [models/gemma4-e4b.md](models/gemma4-e4b.md) | E4B as trained and as exported, its KV caches, and the 512-token rotary blocks in our file |
| [models/gemma4-e2b.md](models/gemma4-e2b.md) | E2B and its differences from E4B, including the artisan `-gpu` files |
| [models/embeddinggemma.md](models/embeddinggemma.md) | EmbeddingGemma 2 text 270m as trained, exported, run and used by Agello |
| [models/tokenizer-and-chat-format.md](models/tokenizer-and-chat-format.md) | The tokenizer, special tokens, chat format, stop tokens and the templates in circulation |
| [generated/](generated/) | Tables produced by the scripts: model files, official configs, binaries, API inventories, flags, kernels by family, our commits |

## Conventions

- **Header block.** Each page opens with its scope (and what it leaves to other pages), the commits
  it is pinned to, the platforms it applies to and the date it was last verified.
- **Citations.** Code links are permalinks made by `tools/agello-docs/cite.py`, never typed by hand.
  Files our commits don't change link to Google's repository at the upstream base; files we change
  link to our fork at the pin. Links between pages are relative. Tables in `generated/` are evidence
  and are linked, not copied.
- **Claim tags and the ledger.** A statement a reader could rely on is a claim. Its tag sits in the
  page source after the sentence (`<!-- c:LM-ENG-042 -->`, invisible when rendered) and points to a
  record in `ledger/*.jsonl` with its sources and the verifier's verdict. Tier-A claims (numbers,
  model architecture, anything that could reach user docs, commit messages or upstream text, and
  anything that contradicts an existing doc) are all verified; tier-B claims are verified on a
  sample, and unsampled ones carry no tag. [maintenance.md](maintenance.md#the-ledger-lifecycle)
  describes the lifecycle.
- **Fact IDs.** `F-NNN` names a row in [facts.md](facts.md) (link `facts.md#f-048`); quote numbers
  from there, with their conditions.
- **"(ours: `<commit>`)"** marks behavior that comes from one of our commits rather than Google's
  code.
- **Which build.** "Documented build" or "our build" means the pins above; "Google's v0.18.0 binary"
  and "the shipped wrapper" mean what Agello ships. A statement about only one of them says so.
- **Open questions.** Anything not verified sits in a page's "Open questions" section, labeled as
  such, rather than in the text.

## Regenerating and re-verifying

- [tools.md](tools.md#toolsagello-docs) lists the scripts in `tools/agello-docs/`: `regen.sh`
  rebuilds every generated table and then runs `check_refs.py`, which checks permalinks, relative
  links and anchors, claim tags, and that each cited range still hashes to the lines the ledger
  recorded.
- [maintenance.md](maintenance.md) gives the procedures: adding a claim or a page, what to do after
  a re-pin or rebase (`check_refs.py --repin` lists the citations to re-verify), after a model update
  and after a new measurement, and who may change what.
