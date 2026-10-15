# T3: the first reply after a helper

Agello runs short helper prompts beside the chat: a title after the first exchange, memories after
a personal message, search queries for a follow-up, document summaries, and dictation. Each runs
in a second `Conversation` on the same engine.

- **Google's v0.18.0** wipes the chat's conversation when the second one runs (Agello probe 1:
  `mainKeptContext` false). So Agello replaced it with an empty one and reseeded the chat from its
  history on the next message, which re-read every token.
- **Our build** (upstream `main` plus our patches) keeps it: the resource manager snapshots the
  chat's KV cache when another session takes the executor, and restores it on the way back
  (probe 1: true).

## P3.1 The cost of a reseed (bench `helper-turn`)

E4B FP16 on the M6, 16K window, MTP on, greedy; a history of filler sentences, a title prompt in a
second conversation, then a short question. The time to first token (TTFT) is the question's.
Build `62b72973cf80` (before P3.3), `results/mac/2026-10-15/bench/t3-helper.jsonl`:

| History | Reseed (what Agello did) | Keep, lock-path snapshots |
| --- | --- | --- |
| 2K tokens | 0.75 s | 0.86–0.88 s |
| 4K tokens | 1.36 s | 0.87–0.89 s |
| 8K tokens | 2.85 s | 0.88–0.90 s |

Keeping the conversation already beat a reseed past 2K, but each switch cost about 290 ms: the
snapshot went through the tensor buffer lock, which converts ML Drift's GPU layout of every INT8
cache to row order on the way out and back on the way in.

## P3.3 Raw snapshots (`agello-next` `aec81cb7`)

When every state buffer has a shared `MTLBuffer` of its own (what `CreateManaged` allocates),
`LitertState::DeepCopy` copies its bytes as they are into a host buffer, and `CopyContentsFrom`
copies them back. No Metal buffer is allocated (b/572018960 is why upstream snapshots to the host).

Same bench, `t3-raw.jsonl` (two runs, MTP on and off):

| History | No helper (control) | Helper, keep |
| --- | --- | --- |
| 2K tokens | 0.11–0.14 s | 0.19–0.21 s |
| 8K tokens | 0.12–0.13 s | 0.16 s |

The reply after the helper is identical to the control's. Exit criterion (≤ 0.3 s at 4K): met.

## P3.2 Agello keeps the chat (Agello `main`)

`LiteRTChatEngine` keeps the chat's conversation through a helper when
`DeviceProfile.keepsConversationAcrossHelpers` allows it:

- the runtime keeps sessions (it reports a build id; Google's binary doesn't), and
- the memory available covers one more copy of the KV cache (28,672 bytes per token of window
  for E4B; twice that before P3.4) above the 400 MB low-memory stop. The Mac doesn't report it,
  so it keeps.

Otherwise it releases the chat's conversation before the helper (so nothing is snapshotted) and
leaves an empty one, as before. `lastHelperKeptConversation` tells `AppState` whether to unbind the
chat; a failed helper reports false. While a helper runs, `cancel()` cancels the helper's
conversation, not the kept one.

Agello on the Mac with our build (Agello `6f4dd45`): unit 226 (1 skipped), inference 21/21,
probes 11/11, thinking 2/2, files 3/3. In `test11_TitleAndContextAfterHelper` (the title helper,
then a recall question) the runtime prefilled 416 tokens instead of 745, and the recall still
names the code word; probe 1 reports `mainKeptContext` true.

## P3.4 Memory: one snapshot, freed at once (`agello-next` `ed998d80`, `585203d3`)

Two costs showed up in the bench after P3.3 (16K window, a 2K history):

1. **A snapshot nothing restores.** After the helper conversation is deleted, its handler stays
   the resource manager's current one until the next switch, which snapshotted the helper's KV
   cache anyway. With the chat's own snapshot waiting to be restored, that held two caches at
   once: +944 MB at the peak. The manager now skips the snapshot of a handler only it holds whose
   processed context no other handler shares (`resource_manager_test`).
2. **A snapshot that stays resident.** The raw snapshot came from `malloc`; freed, its 0.47 GB
   stayed in the allocator's cache, so the footprint after a kept helper stayed 0.47 GB higher
   for the life of the process. Snapshots now live in an anonymous mapping that their deallocator
   unmaps.

| `t3-check-*.jsonl` (16K, MTP on) | P3.3 | + no orphan snapshot | + mapped snapshots |
| --- | --- | --- | --- |
| First reply after a helper, 2K history | 0.187 s | 0.152 s | 0.140 s |
| First reply after a helper, 8K history | 0.157 s | 0.134 s | 0.139 s |
| Peak over the control | +944 MB | +472 MB | +471 MB |
| Footprint after the helper, over the control | +615 MB | +468 MB | 0 |

`t3-cycles-*.jsonl`: 1, 25 and 100 rounds of helper and reply on one conversation. The footprint
doesn't grow with the round count (P3.3: 2,796, 2,909, 2,910 MB, settling after the first rounds;
now 2,348, 2,348, 2,349 MB), so kept conversations don't leak. In `t3-check-*.jsonl` the reply
after the helper is identical to the control's in every build.

So keeping the chat costs one more copy of its KV cache while a helper runs and nothing after.
Agello's policy budgets exactly that (`DeviceProfile.kvBytesPerToken` × window, above the 400 MB
stop): 0.23 GB at the default 8K window, 0.94 GB at 32K.

## Exit criteria

| Criterion (PLAN) | Result |
| --- | --- |
| TTFT after a helper at 4K history ≤ 0.3 s | 0.13–0.15 s (2K and 8K; 4K between) |
| Same greedy continuation | identical to the control in every run |
| No growth over 100 cycles | none |
| A size cap for 8 GB phones | the memory policy above; to be checked on the iPhone |

Still open: an iPhone run (`probes`, `inference`, `memory`) before Agello ships it, and snapshots
proportional to the filled length (a 2K chat in a 16K window still copies the whole cache, which
needs the position stride of ML Drift's KV layout).
