# Draft issue (LiteRT-LM): a failed tool call ends the turn (Swift)

**Title:** Swift: a tool call the model gets wrong throws `toolExecutionError` instead of telling
the model

**Body:**

`Conversation.handleToolCalls` (swift/Conversation.swift) rethrows any error from a tool call as
`toolExecutionError`, which ends the whole turn. Small on-device models get calls wrong often
enough for this to matter:

- a tool name the model invented (`ToolError.notFound`);
- arguments that don't decode into the tool's parameters (`DecodingError`), or arguments given as
  a JSON string holding an object;
- a tool that throws.

The model never sees the failure, so it can't correct the call or answer without the tool.

**What we do:** each failed call gets a `tool_response` of `{"error": "<message>"}` that the model
can act on, and the turn continues. Missing arguments mean an empty object (the tool's defaults
apply) instead of skipping the call, and arguments given as a JSON string holding an object are
decoded. With Gemma 4 E4B, the model then retries with a valid tool or answers directly; the app's
probe for this case (`test15_FailingToolCallKeepsTheReply`) passes on iPhone and Mac.

Patch: https://github.com/PowerBeef/agello-litert-lm/commit/a3c0d0ef729c28d529e2b063785a0808dd19c062
(branch `agello`, swift/Conversation.swift, +55 −12).
