# Draft issue (LiteRT-LM): dropping a conversation during its stream stalls for 10 minutes (Swift)

**Title:** Swift: releasing a Conversation from its own stream callback waits for the engine's
10-minute timeout

**Body:**

`StreamContext` (swift/Conversation.swift) holds its `Conversation` strongly and releases it in
the final stream callback. When the caller has already dropped the conversation (a stream abandoned
early, or a conversation replaced while its cancelled stream finishes), that release is the last
one, so `deinit` runs on LiteRT-LM's callback thread. Deleting the native conversation there waits
for the task that is running the callback, until the engine's timeout:

```
streamCallback -> Conversation.deinit -> ~SessionAdvanced -> WaitUntilSessionDone   (600 s)
```

Seen with v0.18.0's release binary and with a source build of `main`, on iOS and on macOS
(iPhone app on Mac): an app test that drops a voice-turn conversation mid-stream waited 600 s for
its next first token.

**Fix we use:** the stream context holds the conversation weakly, so the conversation is deleted
where the caller drops it, after its stream's work finishes. (Releasing the context on another
queue also ends the deadlock, but lets the deletion overlap the next conversation, which stalled
v0.18.0 the same way in a later turn.) With it, the app's probe suite runs in 367 s instead of
966 s, and the affected probe takes 6 s instead of 600 s.

Patch: https://github.com/PowerBeef/agello-litert-lm/commit/85b1a77695bc90fc8b34a1f455bbe5b2c7fdceff
(branch `agello`, 24 lines in swift/Conversation.swift).
