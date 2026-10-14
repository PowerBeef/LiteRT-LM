# agello-litert-lm

Agello's fork of Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM),
the on-device LLM runtime behind the [Agello](https://github.com/PowerBeef/Agello)
iPhone app. It isn't affiliated with or endorsed by Google.

## Branches

| Branch | What it is |
| --- | --- |
| `agello` (default) | LiteRT-LM **v0.18.0** plus the patches below. Agello vendors its Swift package surface, pinned to a commit. |
| `main` | A mirror of upstream `main`. |
| `research/3913` | The investigation and model-side fix for [LiteRT-LM #3913](https://github.com/google-ai-edge/LiteRT-LM/issues/3913) (float16 rotary positions on the GPU). |

## Patches on `agello`

1. **Failed tool calls don't end replies** (`swift/Conversation.swift`).
   v0.18.0 rethrows any tool-call failure (a tool name the model invents,
   arguments that don't decode, a tool that throws) as `toolExecutionError`,
   and the turn ends. Here each failed call is answered with an `error` tool
   response the model can read, so it can fix the call or answer without it.

## What isn't changed

The native runtime. Agello links Google's checksummed `CLiteRTLM.xcframework`
release binary, as `Package.swift` does upstream, and the Metal GPU
accelerator is a prebuilt binary even upstream. The patches here are to the
Swift wrapper.

## Updating

1. `git fetch upstream --tags`, then `git rebase --onto vX.Y.Z v0.18.0 agello`
   and resolve the patches.
2. Cherry-pick single upstream fixes with `git cherry-pick <commit>`, as long
   as they work with the release binary in use.
3. Push, then vendor the new commit in Agello:
   `REF=<commit> ./Vendor/refresh-litert-lm.sh`.

Upstream's own README is [`README.md`](../README.md).

## License

Apache 2.0, like upstream ([`LICENSE`](../LICENSE)). Modified files carry a
notice that describes the change.
