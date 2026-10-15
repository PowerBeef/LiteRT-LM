# Draft issue (LiteRT-LM): session switches on Metal cost 2× the KV cache and ~0.3 s per switch

**Title:** Metal: a second conversation (session switch) snapshots the KV caches through layout
conversion, snapshots deleted sessions, and leaves the snapshot resident

**Body:**

Apps run short helper prompts beside a chat (titles, memory extraction, query planning) in a
second `Conversation` on the same engine. On `main`, the resource manager snapshots the chat's KV
caches when the helper takes the executor and restores them afterwards (v0.18.0's release binary
wiped the chat instead). On Metal, three things make that expensive. Numbers are for Gemma 4 E4B,
FP16, a 16K window, on an M6 Mac:

1. **Layout conversion.** `LitertState::DeepCopy` copies each state buffer through
   `CopyTensorBuffer`, whose lock converts ML Drift's GPU layout to row order (a transpose of the
   INT8 caches), and `CopyContentsFrom` converts it back: about 290 ms per switch. Every state
   buffer is a shared `MTLBuffer` of its own (what `CreateManaged` allocates), so its bytes can be
   copied as they are into a host buffer and back. No Metal buffer is allocated, which keeps the
   b/572018960 workaround intact. The switch then takes 44–156 ms.
2. **Snapshots of deleted sessions.** After the helper conversation is deleted, its handler stays
   the resource manager's `current_handler_` (held only there) until the next switch, which
   snapshots it anyway: a full KV copy that nothing restores, and twice the KV cache in memory
   while the chat's snapshot comes back. Skipping the snapshot when `current_handler_` is the only
   owner and no other handler shares its processed context takes the peak from +944 MB to +472 MB.
3. **Resident snapshots.** A 0.47 GB snapshot from `malloc` stays in the allocator's cache after
   it's freed, so the process footprint stays 0.47 GB higher after the first helper. Mapping
   snapshots anonymously and unmapping them in the deallocator returns the footprint to its level
   before the helper.

Result: the first reply after a helper comes 0.13–0.15 s after the prompt, against 0.75–2.85 s to
re-read 2K–8K tokens of history and 0.11–0.13 s with no helper at all. The reply is identical to
the reply without the helper, and 100 helper rounds on one conversation leave the footprint flat.

Patches (branch `agello-next` of https://github.com/PowerBeef/agello-litert-lm):

- `aec81cb7` State: raw snapshots of Metal KV caches for session switches
  (`runtime/util/metal_raw_buffer.{h,mm}`, `state.cc`);
- `ed998d80` Sessions: no snapshot of a deleted session on a switch (`resource_manager.cc`, a test);
- `585203d3` State: raw snapshots in pages of their own.

Further: snapshots could copy only the filled positions (a 2K chat in a 16K window still copies
the whole cache), or stay on the GPU once b/572018960 allows Metal allocations there.
