// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Modified by Patrice Déry for Agello (PowerBeef/agello-litert-lm, branch agello):
// a tool call that fails is answered with an error response the model can
// read, instead of throwing and ending the turn (see handleToolCalls); a
// stream holds its conversation weakly, so the last reference is never
// released inside LiteRT-LM's callback, where deleting the conversation
// deadlocked (see StreamContext); and streams are found by ID instead of a
// retained pointer, end once however many final chunks LiteRT-LM sends,
// stop their native work when cancelled (between tool rounds too), and cap
// their tool rounds (see ConversationStream.swift).

import Foundation
import OSLog
import CLiteRTLM

extension ResponseFormat.FormatType {
  var cConstraintType: LiteRtLmConstraintType {
    switch self {
    case .regex:
      return kLiteRtLmConstraintTypeRegex
    case .jsonObject:
      return kLiteRtLmConstraintTypeJsonSchema
    }
  }
}

typealias CConversationHandle = OpaquePointer

private let logger = Logger(
  subsystem: "com.google.odml.litertlm.swift",
  category: "Conversation"
)

private let recurringToolCallLimit = 25

/// Represents a conversation with the LiteRT-LM model.
///
/// Example usage:
/// ```swift
/// // Assuming 'engine' is an instance of Engine
/// let conversation = try await engine.createConversation()
///
/// // Send a message and get the response.
/// let response = try conversation.sendMessage(Message("Hello world"))
///
/// // Send a message async with response chunks as AsyncThrowingStream.
/// for try await chunk in conversation.sendMessageStream(Message("Hello world")) {
///   print(chunk.text)
/// }
/// ```
///
/// This class facilitates interaction with the LiteRT-LM model by handling message sending
/// and response reception.
public final class Conversation: Sendable {
  private let logger = Logger(
    subsystem: "com.google.ai.edge.litertlm.swift",
    category: "Conversation"
  )

  private let handle: CConversationHandle?
  private let toolManager: ToolManager
  private let automaticToolCalling: Bool
  private let engine: Engine
  private let enableResponseFormat: Bool
  private let visualTokenBudget: Int32?
  /// Identifies this conversation's streams in ``StreamTable``, after it's gone too.
  let streamOwner = StreamTable.shared.makeOwner()

  /// Whether the conversation is alive and ready to be used.
  public var isAlive: Bool {
    return handle != nil
  }

  init(
    handle: CConversationHandle, toolManager: ToolManager, automaticToolCalling: Bool = true,
    engine: Engine, enableResponseFormat: Bool = false, visualTokenBudget: Int32? = nil
  ) {
    self.handle = handle
    self.toolManager = toolManager
    self.automaticToolCalling = automaticToolCalling
    self.engine = engine
    self.enableResponseFormat = enableResponseFormat
    self.visualTokenBudget = visualTokenBudget
  }

  deinit {
    if let handle = handle {
      litert_lm_conversation_delete(handle)
    }
    // Deleting waits for running sends; any stream still registered gets no more chunks.
    for context in StreamTable.shared.retireAll(owner: streamOwner) {
      context.finish(throwing: LiteRTLMError.conversation(.notAlive))
    }
  }

  private func shouldApplyResponseFormat(
    _ responseFormat: ResponseFormat?,
    forMessageDict messageDict: [String: Any]
  ) -> Bool {
    guard responseFormat != nil else { return false }
    let isToolResponse = (messageDict["role"] as? String) == "tool"
    if automaticToolCalling && toolManager.toolsJsonDescription != "[]" && !isToolResponse {
      return false
    }
    return true
  }

  /// Sends a message to the model and returns the response. This is a synchronous call.
  ///
  /// - Parameter message: The message to send to the model.
  /// - Parameter extraContext: The extra context to send to the model.
  /// - Parameter repetitionPenaltyConfig: Optional configuration for repetition penalty.
  /// - Parameter noRepeatNgramConfig: Optional configuration for no repeat ngram penalty.
  /// - Parameter maxOutputTokens: Optional maximum number of tokens to generate per response. For thinking models, both thinking (reasoning) tokens and the final response tokens count towards this limit.
  /// - Parameter thinkingConfig: Optional configuration for thinking/reasoning generation.
  /// - Parameter responseFormat: Optional response format for constrained decoding.
  /// - Returns: The model's response message.
  /// - Throws: A `LiteRTLMError` if sending the message fails or the model
  ///   returns an invalid response.
  public func sendMessage(
    _ message: Message,
    extraContext: [String: Any]? = nil,
    repetitionPenaltyConfig: RepetitionPenaltyConfig? = nil,
    noRepeatNgramConfig: NoRepeatNgramConfig? = nil,
    suppressTokensConfig: SuppressTokensConfig? = nil,
    maxOutputTokens: Int? = nil,
    thinkingConfig: ThinkingConfig? = nil,
    responseFormat: ResponseFormat? = nil
  ) async throws -> Message {
    let handle = try checkIsAlive()

    if responseFormat != nil && !enableResponseFormat {
      throw LiteRTLMError.conversation(.responseFormatNotEnabled)
    }

    var currentMessageJson: [String: Any] = message.toJson

    for i in 0..<recurringToolCallLimit {
      let (responseJson, responseString) = try attemptSendMessage(
        handle: handle,
        messageJson: currentMessageJson,
        extraContext: extraContext,
        repetitionPenaltyConfig: repetitionPenaltyConfig,
        noRepeatNgramConfig: noRepeatNgramConfig,
        suppressTokensConfig: suppressTokensConfig,
        maxOutputTokens: maxOutputTokens,
        thinkingConfig: i == 0 ? thinkingConfig : nil,
        responseFormat: responseFormat
      )

      guard let toolCalls = responseJson["tool_calls"] as? [[String: Any]] else {
        if responseJson["content"] != nil || responseJson["channels"] != nil {
          return try Conversation.jsonToMessage(responseString)
        } else {
          throw LiteRTLMError.conversation(.invalidResponse(responseString))
        }
      }
      if !automaticToolCalling {
        return try Conversation.jsonToMessage(responseString)
      }
      currentMessageJson = try await handleToolCalls(toolCalls)
    }
    throw LiteRTLMError.conversation(
      .recurringToolCallLimitExceeded(limit: recurringToolCallLimit))
  }

  private func attemptSendMessage(
    handle: CConversationHandle,
    messageJson: [String: Any],
    extraContext: [String: Any]?,
    repetitionPenaltyConfig: RepetitionPenaltyConfig? = nil,
    noRepeatNgramConfig: NoRepeatNgramConfig? = nil,
    suppressTokensConfig: SuppressTokensConfig? = nil,
    maxOutputTokens: Int? = nil,
    thinkingConfig: ThinkingConfig? = nil,
    responseFormat: ResponseFormat? = nil
  ) throws
    -> (responseJson: [String: Any], responseString: String)
  {
    let messageData = try JSONSerialization.data(withJSONObject: messageJson)
    guard let messageString = String(data: messageData, encoding: .utf8) else {
      throw LiteRTLMError.conversation(.failedToSerializeMessage)
    }

    var extraContextString: String? = nil
    if let extraContext = extraContext, !extraContext.isEmpty,
      let extraData = try? JSONSerialization.data(withJSONObject: extraContext)
    {
      extraContextString = String(data: extraData, encoding: .utf8)
    }
    let optionalArgs = try makeOptionalArgs(
      forMessageDict: messageJson,
      repetitionPenaltyConfig: repetitionPenaltyConfig,
      noRepeatNgramConfig: noRepeatNgramConfig,
      suppressTokensConfig: suppressTokensConfig,
      maxOutputTokens: maxOutputTokens,
      thinkingConfig: thinkingConfig,
      responseFormat: responseFormat
    )
    defer { litert_lm_conversation_optional_args_delete(optionalArgs) }

    let responseError: (String) -> LiteRTLMError = { .conversation(.invalidResponse($0)) }
    let responsePtr = try LiteRTLMError.create(
      "litert_lm_conversation_send_message", responseError
    ) { out in
      litert_lm_conversation_send_message(
        handle, messageString, extraContextString, optionalArgs, out)
    }
    // Delete the response pointer at the end of each iteration. Handled by defer block.
    let responsePtrRef = responsePtr
    defer { litert_lm_json_response_delete(responsePtrRef) }

    let responseChars: UnsafePointer<CChar>? = try LiteRTLMError.get(
      nil, "litert_lm_json_response_get_string", responseError
    ) { litert_lm_json_response_get_string(responsePtr, $0) }
    guard let responseChars else {
      throw LiteRTLMError.conversation(
        .invalidResponse("litert_lm_json_response_get_string returned a null string"))
    }
    let responseString = String(cString: responseChars)

    guard let responseData = responseString.data(using: .utf8),
      let responseJson = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]
    else {
      throw LiteRTLMError.conversation(.invalidJson("Failed to parse native response JSON."))
    }
    return (responseJson, responseString)
  }

  /// Creates native optional args for a single `send_message` call and applies the given settings.
  ///
  /// - Parameters:
  ///   - messageJson: The message being sent, used to decide whether `responseFormat` applies.
  ///   - repetitionPenaltyConfig: Optional configuration for repetition penalty.
  ///   - noRepeatNgramConfig: Optional configuration for no-repeat n-gram.
  ///   - suppressTokensConfig: Optional configuration for suppressed tokens.
  ///   - maxOutputTokens: Optional maximum number of output tokens.
  ///   - thinkingConfig: Optional configuration for thinking/reasoning generation.
  ///   - responseFormat: Optional response format for constrained decoding.
  /// - Returns: The native optional args handle. The caller must delete it with
  ///   `litert_lm_conversation_optional_args_delete`.
  /// - Throws: A `LiteRTLMError` if creating the optional args or any native setter fails. The
  ///   optional args are deleted before the error is thrown.
  private func makeOptionalArgs(
    forMessageDict messageJson: [String: Any],
    repetitionPenaltyConfig: RepetitionPenaltyConfig?,
    noRepeatNgramConfig: NoRepeatNgramConfig?,
    suppressTokensConfig: SuppressTokensConfig?,
    maxOutputTokens: Int?,
    thinkingConfig: ThinkingConfig?,
    responseFormat: ResponseFormat?
  ) throws -> OpaquePointer {
    let optionalArgsError: (String) -> LiteRTLMError = { .conversation(.invalidResponse($0)) }
    let optionalArgs = try LiteRTLMError.create(
      "litert_lm_conversation_optional_args_create", optionalArgsError
    ) { out in
      litert_lm_conversation_optional_args_create(out)
    }
    do {
      if let visualTokenBudget = self.visualTokenBudget ?? ExperimentalFlags.visualTokenBudget {
        try LiteRTLMError.check(
          litert_lm_conversation_optional_args_set_visual_token_budget(
            optionalArgs, Int32(visualTokenBudget)),
          "litert_lm_conversation_optional_args_set_visual_token_budget", optionalArgsError)
      }
      if let repetitionPenaltyConfig = repetitionPenaltyConfig {
        let cRepetitionPenaltyConfig = try LiteRTLMError.create(
          "litert_lm_repetition_penalty_config_create", optionalArgsError
        ) { out in
          litert_lm_repetition_penalty_config_create(out)
        }
        defer { litert_lm_repetition_penalty_config_delete(cRepetitionPenaltyConfig) }

        if let repetitionPenalty = repetitionPenaltyConfig.repetitionPenalty {
          try LiteRTLMError.check(
            litert_lm_repetition_penalty_config_set_repetition_penalty(
              cRepetitionPenaltyConfig, repetitionPenalty),
            "litert_lm_repetition_penalty_config_set_repetition_penalty", optionalArgsError)
        }
        if let presencePenalty = repetitionPenaltyConfig.presencePenalty {
          try LiteRTLMError.check(
            litert_lm_repetition_penalty_config_set_presence_penalty(
              cRepetitionPenaltyConfig, presencePenalty),
            "litert_lm_repetition_penalty_config_set_presence_penalty", optionalArgsError)
        }
        if let frequencyPenalty = repetitionPenaltyConfig.frequencyPenalty {
          try LiteRTLMError.check(
            litert_lm_repetition_penalty_config_set_frequency_penalty(
              cRepetitionPenaltyConfig, frequencyPenalty),
            "litert_lm_repetition_penalty_config_set_frequency_penalty", optionalArgsError)
        }
        if let windowSize = repetitionPenaltyConfig.windowSize {
          try LiteRTLMError.check(
            litert_lm_repetition_penalty_config_set_window_size(
              cRepetitionPenaltyConfig, Int32(windowSize)),
            "litert_lm_repetition_penalty_config_set_window_size", optionalArgsError)
        }
        try LiteRTLMError.check(
          litert_lm_conversation_optional_args_set_repetition_penalty_config(
            optionalArgs, cRepetitionPenaltyConfig),
          "litert_lm_conversation_optional_args_set_repetition_penalty_config", optionalArgsError)
      }
      if let noRepeatNgramConfig = noRepeatNgramConfig {
        let cNoRepeatNgramConfig = try LiteRTLMError.create(
          "litert_lm_no_repeat_ngram_config_create", optionalArgsError
        ) { out in
          litert_lm_no_repeat_ngram_config_create(out)
        }
        defer { litert_lm_no_repeat_ngram_config_delete(cNoRepeatNgramConfig) }

        if let noRepeatNgramSize = noRepeatNgramConfig.noRepeatNgramSize {
          try LiteRTLMError.check(
            litert_lm_no_repeat_ngram_config_set_no_repeat_ngram_size(
              cNoRepeatNgramConfig, Int32(noRepeatNgramSize)),
            "litert_lm_no_repeat_ngram_config_set_no_repeat_ngram_size", optionalArgsError)
        }
        if let windowSize = noRepeatNgramConfig.windowSize {
          try LiteRTLMError.check(
            litert_lm_no_repeat_ngram_config_set_window_size(
              cNoRepeatNgramConfig, Int32(windowSize)),
            "litert_lm_no_repeat_ngram_config_set_window_size", optionalArgsError)
        }
        try LiteRTLMError.check(
          litert_lm_conversation_optional_args_set_no_repeat_ngram_config(
            optionalArgs, cNoRepeatNgramConfig),
          "litert_lm_conversation_optional_args_set_no_repeat_ngram_config", optionalArgsError)
      }
      if let suppressTokens = suppressTokensConfig?.suppressTokens, !suppressTokens.isEmpty {
        let cSuppressTokensConfig = try LiteRTLMError.create(
          "litert_lm_suppress_tokens_config_create", optionalArgsError
        ) { out in
          litert_lm_suppress_tokens_config_create(out)
        }
        defer { litert_lm_suppress_tokens_config_delete(cSuppressTokensConfig) }

        let cTokens = suppressTokens.map { Int32($0) }
        let suppressTokensStatus = cTokens.withUnsafeBufferPointer { buffer in
          litert_lm_suppress_tokens_config_set_suppress_tokens(
            cSuppressTokensConfig, buffer.baseAddress, buffer.count)
        }
        try LiteRTLMError.check(
          suppressTokensStatus, "litert_lm_suppress_tokens_config_set_suppress_tokens",
          optionalArgsError)
        try LiteRTLMError.check(
          litert_lm_conversation_optional_args_set_suppress_tokens_config(
            optionalArgs, cSuppressTokensConfig),
          "litert_lm_conversation_optional_args_set_suppress_tokens_config", optionalArgsError)
      }
      if let maxOutputTokens = maxOutputTokens {
        try LiteRTLMError.check(
          litert_lm_conversation_optional_args_set_max_output_tokens(
            optionalArgs, Int32(maxOutputTokens)),
          "litert_lm_conversation_optional_args_set_max_output_tokens", optionalArgsError)
      }
      if let thinkingConfig = thinkingConfig {
        let cThinkingConfig = try LiteRTLMError.create(
          "litert_lm_thinking_config_create", optionalArgsError
        ) { out in
          litert_lm_thinking_config_create(out)
        }
        defer { litert_lm_thinking_config_delete(cThinkingConfig) }
        try LiteRTLMError.check(
          litert_lm_thinking_config_set_enable_thinking(
            cThinkingConfig, thinkingConfig.enableThinking),
          "litert_lm_thinking_config_set_enable_thinking", optionalArgsError)
        try LiteRTLMError.check(
          litert_lm_thinking_config_set_thinking_token_budget(
            cThinkingConfig, Int32(thinkingConfig.thinkingTokenBudget)),
          "litert_lm_thinking_config_set_thinking_token_budget", optionalArgsError)
        try LiteRTLMError.check(
          litert_lm_conversation_optional_args_set_thinking_config(optionalArgs, cThinkingConfig),
          "litert_lm_conversation_optional_args_set_thinking_config", optionalArgsError)
      }
      if let responseFormat = responseFormat,
        shouldApplyResponseFormat(responseFormat, forMessageDict: messageJson)
      {
        try LiteRTLMError.check(
          litert_lm_conversation_optional_args_set_constraint(
            optionalArgs, responseFormat.type.cConstraintType, responseFormat.schemaOrPattern),
          "litert_lm_conversation_optional_args_set_constraint", optionalArgsError)
      }
      return optionalArgs
    } catch {
      litert_lm_conversation_optional_args_delete(optionalArgs)
      throw error
    }
  }

  /// Runs the model's tool calls. A call that fails (a tool name the model
  /// invented, arguments that don't decode into the tool's parameters, or a
  /// tool that throws) is answered with an `error` response, so the model
  /// can correct the call or answer without it instead of the turn ending.
  fileprivate func handleToolCalls(_ toolCalls: [[String: Any]]) async throws -> [String: Any] {
    var toolResponses: [[String: Any]] = []

    for toolCall in toolCalls {
      guard let function = toolCall["function"] as? [String: Any],
        let name = function["name"] as? String
      else {
        continue
      }
      let response: Any
      if let argsObject = Conversation.toolArguments(function["arguments"]) {
        do {
          response = try await toolManager.execute(name: name, arguments: argsObject)
        } catch {
          logger.warning("Tool call \(name, privacy: .public) failed: \(String(describing: error), privacy: .public)")
          response = ["error": Conversation.toolErrorMessage(name: name, error: error)]
        }
      } else {
        response = ["error": "The arguments for '\(name)' must be a JSON object."]
      }
      toolResponses.append([
        "type": "tool_response",
        "name": name,
        "response": response,
      ])
    }

    return ["role": "tool", "content": toolResponses]
  }

  /// A tool call's arguments as an object: missing means none, and a JSON
  /// string holding an object is decoded. Anything else is invalid (nil).
  fileprivate static func toolArguments(_ value: Any?) -> [String: Any]? {
    switch value {
    case nil, is NSNull:
      return [:]
    case let object as [String: Any]:
      return object
    case let text as String:
      guard let data = text.data(using: .utf8),
        let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
      else {
        return nil
      }
      return object
    default:
      return nil
    }
  }

  /// What the model reads when a tool call fails.
  fileprivate static func toolErrorMessage(name: String, error: Error) -> String {
    if case LiteRTLMError.tool(.notFound) = error {
      return "There is no tool named '\(name)'. Use only the tools you were given, or answer without one."
    }
    if error is DecodingError {
      return "The arguments for '\(name)' don't match its parameters. Check their names and types, then try again."
    }
    return "'\(name)' failed: \(error.localizedDescription)"
  }

  /// Throws an error if the conversation is not alive.
  ///
  /// - Returns: The `OpaquePointer` handle if the conversation is alive.
  /// - Throws: A `LiteRTLMError` if `handle` is nil, indicating the conversation is not alive.
  fileprivate func checkIsAlive() throws -> OpaquePointer {
    guard let handle else {
      throw LiteRTLMError.conversation(.notAlive)
    }
    return handle
  }

  /// Sends a message to the model and returns an async stream of response chunks.
  ///
  /// - Parameter message: The message to send.
  /// - Parameter extraContext: The extra context to send to the model.
  /// - Parameter repetitionPenaltyConfig: Optional configuration for repetition penalty.
  /// - Parameter maxOutputTokens: Optional maximum number of tokens to generate per response. For thinking models, both thinking (reasoning) tokens and the final response tokens count towards this limit.
  /// - Parameter thinkingConfig: Optional configuration for thinking/reasoning generation.
  /// - Parameter responseFormat: Optional response format for constrained decoding.
  /// - Parameter followUpMaxOutputTokens: Optional output cap for each generation after a tool
  ///   response; `maxOutputTokens` when nil.
  /// - Parameter maxToolRounds: Optional limit on automatic tool rounds; past it the stream ends
  ///   with `recurringToolCallLimitExceeded`. 25 when nil.
  /// - Returns: An async throwing stream of `Message` chunks. Cancelling the task that iterates
  ///   it stops the native work, which leaves the conversation cancelled, like `cancel()`.
  public func sendMessageStream(
    _ message: Message,
    extraContext: [String: Any]? = nil,
    repetitionPenaltyConfig: RepetitionPenaltyConfig? = nil,
    noRepeatNgramConfig: NoRepeatNgramConfig? = nil,
    suppressTokensConfig: SuppressTokensConfig? = nil,
    maxOutputTokens: Int? = nil,
    thinkingConfig: ThinkingConfig? = nil,
    responseFormat: ResponseFormat? = nil,
    followUpMaxOutputTokens: Int? = nil,
    maxToolRounds: Int? = nil
  ) -> AsyncThrowingStream<Message, Error> {
    return AsyncThrowingStream { continuation in
      let options = StreamOptions(
        repetitionPenaltyConfig: repetitionPenaltyConfig,
        noRepeatNgramConfig: noRepeatNgramConfig,
        suppressTokensConfig: suppressTokensConfig,
        maxOutputTokens: maxOutputTokens,
        followUpMaxOutputTokens: followUpMaxOutputTokens,
        responseFormat: responseFormat,
        maxToolRounds: maxToolRounds ?? recurringToolCallLimit
      )
      let effects = StreamEffects(
        cancelNative: { [weak self] in try? self?.cancel() },
        startToolRound: { context, batch in
          guard let conversation = context.conversation else {
            throw LiteRTLMError.conversation(.notAlive)
          }
          try await conversation.runToolRound(batch.calls, context: context)
        }
      )
      let context = StreamContext(
        continuation: continuation, conversation: self, owner: streamOwner, options: options,
        effects: effects)
      context.cancelNativeWorkWhenAbandoned()
      do {
        if responseFormat != nil && !enableResponseFormat {
          throw LiteRTLMError.conversation(.responseFormatNotEnabled)
        }
        let handle = try self.checkIsAlive()
        let messageJson: [String: Any] = message.toJson
        try self.sendToStream(
          handle: handle,
          messageJson: messageJson,
          extraContext: extraContext,
          repetitionPenaltyConfig: repetitionPenaltyConfig,
          noRepeatNgramConfig: noRepeatNgramConfig,
          suppressTokensConfig: suppressTokensConfig,
          maxOutputTokens: maxOutputTokens,
          thinkingConfig: thinkingConfig,
          responseFormat: responseFormat,
          context: context
        )
      } catch {
        context.finish(throwing: error)
      }
    }
  }

  /// Runs a round's tool calls and sends their responses, as the stream's next round.
  fileprivate func runToolRound(_ toolCalls: [[String: Any]], context: StreamContext) async throws
  {
    let toolResponseJson = try await handleToolCalls(toolCalls)
    // A Stop while the tools ran found no native task to cancel.
    if context.isCancelled { throw CancellationError() }
    try sendToStream(
      handle: checkIsAlive(),
      messageJson: toolResponseJson,
      repetitionPenaltyConfig: context.options.repetitionPenaltyConfig,
      noRepeatNgramConfig: context.options.noRepeatNgramConfig,
      suppressTokensConfig: context.options.suppressTokensConfig,
      maxOutputTokens: context.options.followUpMaxOutputTokens ?? context.options.maxOutputTokens,
      responseFormat: context.options.responseFormat,
      context: context
    )
  }

  /// Sends a message to the model and handles the response via a streaming callback.
  ///
  /// This function is used internally by `sendMessageStream` and for handling
  /// subsequent tool call responses within the stream.
  ///
  /// - Parameters:
  ///   - handle: The `CConversationHandle` for the current conversation.
  ///   - messageJson: The message to send, represented as a JSON dictionary.
  ///   - extraContext: The extra context to send to the model.
  ///   - repetitionPenaltyConfig: Optional configuration for repetition penalty.
  ///   - thinkingConfig: Optional configuration for thinking/reasoning generation.
  ///   - maxOutputTokens: Optional maximum number of output tokens.
  ///   - context: The `StreamContext` containing the `AsyncThrowingStream.Continuation`
  ///     and other state for the streaming process.
  /// - Throws: A `LiteRTLMError` if the message fails to send or the response is invalid.
  ///   native `send_message_stream` call fails.
  func sendToStream(
    handle: CConversationHandle,
    messageJson: [String: Any],
    extraContext: [String: Any]? = nil,
    repetitionPenaltyConfig: RepetitionPenaltyConfig? = nil,
    noRepeatNgramConfig: NoRepeatNgramConfig? = nil,
    suppressTokensConfig: SuppressTokensConfig? = nil,
    maxOutputTokens: Int? = nil,
    thinkingConfig: ThinkingConfig? = nil,
    responseFormat: ResponseFormat? = nil,
    context: StreamContext
  ) throws {
    let messageData = try JSONSerialization.data(withJSONObject: messageJson)
    guard let messageString = String(data: messageData, encoding: .utf8) else {
      throw LiteRTLMError.conversation(.failedToSerializeMessage)
    }

    var extraContextString: String? = nil
    if let extraContext = extraContext, !extraContext.isEmpty,
      let extraData = try? JSONSerialization.data(withJSONObject: extraContext)
    {
      extraContextString = String(data: extraData, encoding: .utf8)
    }

    let optionalArgs = try makeOptionalArgs(
      forMessageDict: messageJson,
      repetitionPenaltyConfig: repetitionPenaltyConfig,
      noRepeatNgramConfig: noRepeatNgramConfig,
      suppressTokensConfig: suppressTokensConfig,
      maxOutputTokens: maxOutputTokens,
      thinkingConfig: thinkingConfig,
      responseFormat: responseFormat
    )
    defer { litert_lm_conversation_optional_args_delete(optionalArgs) }

    // Registered before the call: LiteRT-LM can call back before it returns.
    let id = StreamTable.shared.register(context)
    let status = litert_lm_conversation_send_message_stream(
      handle,
      messageString,
      extraContextString,
      optionalArgs,
      streamCallback,
      UnsafeMutableRawPointer(bitPattern: id)
    )

    guard status == kLiteRtLmStatusOk else {
      StreamTable.shared.retire(id)
      let errorMsg = LiteRTLMError.consumeLastError() ?? ""
      throw LiteRTLMError.conversation(
        .failedToStartStream(status: Int(status.rawValue), message: errorMsg))
    }
    // A Stop that came while this round was being set up had no native task to cancel yet.
    if context.isCancelled {
      _ = litert_lm_conversation_cancel_process(handle)
    }
  }

  /// Cancels the ongoing asynchronous inference process, and any stream between tool rounds.
  public func cancel() throws {
    let handle = try checkIsAlive()
    for context in StreamTable.shared.contexts(owner: streamOwner) {
      context.markCancelled()
    }
    try LiteRTLMError.check(
      litert_lm_conversation_cancel_process(handle), "litert_lm_conversation_cancel_process",
      { .conversation(.invalidResponse($0)) })
  }

  /// Renders the message into a string for testing and logging.
  ///
  /// This function does not need to be called for actual message sending, as the `sendMessage` and
  /// `sendMessageStream` functions will handle rendering internally.
  ///
  /// - Parameter message: The message to render.
  /// - Returns: The rendered message string.
  /// - Throws: A `LiteRTLMError` if the conversation is not alive, serializing fails, or rendering fails.
  public func renderMessageIntoString(_ message: Message) throws -> String {
    let handle = try checkIsAlive()
    let messageData = try JSONSerialization.data(withJSONObject: message.toJson)
    guard let messageString = String(data: messageData, encoding: .utf8) else {
      throw LiteRTLMError.conversation(.failedToSerializeMessage)
    }

    let renderError: (String) -> LiteRTLMError = { .conversation(.invalidResponse($0)) }
    let cString: UnsafePointer<CChar>? = try LiteRTLMError.get(
      nil, "litert_lm_conversation_render_message_to_string", renderError
    ) { litert_lm_conversation_render_message_to_string(handle, messageString, $0) }
    guard let cString else {
      throw LiteRTLMError.conversation(
        .invalidResponse("litert_lm_conversation_render_message_to_string returned a null string"))
    }
    return String(cString: cString)
  }

  /// Renders the preface into a string for testing and logging.
  ///
  /// - Returns: The rendered preface string.
  /// - Throws: A `LiteRTLMError` if the conversation is not alive, or rendering fails.
  public func renderPrefaceIntoString() throws -> String {
    let handle = try checkIsAlive()
    let renderError: (String) -> LiteRTLMError = { .conversation(.invalidResponse($0)) }
    let cString: UnsafePointer<CChar>? = try LiteRTLMError.get(
      nil, "litert_lm_conversation_render_preface_to_string", renderError
    ) { litert_lm_conversation_render_preface_to_string(handle, $0) }
    guard let cString else {
      throw LiteRTLMError.conversation(
        .invalidResponse("litert_lm_conversation_render_preface_to_string returned a null string"))
    }
    return String(cString: cString)
  }

  /// Gets the number of tokens in the conversation KV Cache (prefill + decode).
  ///
  /// - Throws: A `LiteRTLMError` if the conversation is not alive or the native call fails.
  public func getTokenCount() throws -> Int {
    let handle = try checkIsAlive()
    let tokenCount = try LiteRTLMError.get(
      Int32(0), "litert_lm_conversation_get_token_count",
      { .conversation(.invalidResponse($0)) }
    ) { litert_lm_conversation_get_token_count(handle, $0) }
    return Int(tokenCount)
  }

  /// Retrieves the benchmark information from the conversation.
  ///
  /// - Returns: The benchmark information
  /// - Throws: A `LiteRTLMError` if the benchmark flag is not enabled or info is unavailable.
  public func getBenchmarkInfo() throws -> BenchmarkInfo {
    let handle = try checkIsAlive()

    if !ExperimentalFlags.enableBenchmark {
      throw LiteRTLMError.conversation(.benchmarkNotEnabled)
    }

    let makeError: (String) -> LiteRTLMError = { details in
      self.logger.error("Failed to read benchmark info: \(details)")
      return LiteRTLMError.conversation(.benchmarkInfoUnavailable)
    }

    let benchmarkInfoPtr = try LiteRTLMError.create(
      "litert_lm_conversation_get_benchmark_info", makeError
    ) { out in
      litert_lm_conversation_get_benchmark_info(handle, out)
    }
    defer { litert_lm_benchmark_info_delete(benchmarkInfoPtr) }

    let numPrefillTurns = try LiteRTLMError.get(
      Int32(0), "litert_lm_benchmark_info_get_num_prefill_turns", makeError
    ) { litert_lm_benchmark_info_get_num_prefill_turns(benchmarkInfoPtr, $0) }
    let numDecodeTurns = try LiteRTLMError.get(
      Int32(0), "litert_lm_benchmark_info_get_num_decode_turns", makeError
    ) { litert_lm_benchmark_info_get_num_decode_turns(benchmarkInfoPtr, $0) }

    let initTimeInSecond = try LiteRTLMError.get(
      0.0, "litert_lm_benchmark_info_get_total_init_time_in_second", makeError
    ) { litert_lm_benchmark_info_get_total_init_time_in_second(benchmarkInfoPtr, $0) }
    let timeToFirstTokenInSecond = try LiteRTLMError.get(
      0.0, "litert_lm_benchmark_info_get_time_to_first_token", makeError
    ) { litert_lm_benchmark_info_get_time_to_first_token(benchmarkInfoPtr, $0) }

    var lastPrefillTokenCount = 0
    var lastPrefillTokensPerSec = 0.0
    if numPrefillTurns > 0 {
      lastPrefillTokenCount = try Int(
        LiteRTLMError.get(
          Int32(0), "litert_lm_benchmark_info_get_prefill_token_count_at", makeError
        ) {
          litert_lm_benchmark_info_get_prefill_token_count_at(
            benchmarkInfoPtr, numPrefillTurns - 1, $0)
        })
      lastPrefillTokensPerSec = try LiteRTLMError.get(
        0.0, "litert_lm_benchmark_info_get_prefill_tokens_per_sec_at", makeError
      ) {
        litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
          benchmarkInfoPtr, numPrefillTurns - 1, $0)
      }
    }

    var lastDecodeTokenCount = 0
    var lastDecodeTokensPerSec = 0.0
    if numDecodeTurns > 0 {
      lastDecodeTokenCount = try Int(
        LiteRTLMError.get(
          Int32(0), "litert_lm_benchmark_info_get_decode_token_count_at", makeError
        ) {
          litert_lm_benchmark_info_get_decode_token_count_at(
            benchmarkInfoPtr, numDecodeTurns - 1, $0)
        })
      lastDecodeTokensPerSec = try LiteRTLMError.get(
        0.0, "litert_lm_benchmark_info_get_decode_tokens_per_sec_at", makeError
      ) {
        litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
          benchmarkInfoPtr, numDecodeTurns - 1, $0)
      }
    }

    return BenchmarkInfo(
      initTimeInSecond: initTimeInSecond,
      timeToFirstTokenInSecond: timeToFirstTokenInSecond,
      lastPrefillTokenCount: lastPrefillTokenCount,
      lastDecodeTokenCount: lastDecodeTokenCount,
      lastPrefillTokensPerSecond: lastPrefillTokensPerSec,
      lastDecodeTokensPerSecond: lastDecodeTokensPerSec
    )
  }

  /// Internal Helper Function to convert a JSON string to a `Message`.
  ///
  /// - Parameter jsonString: The JSON string to convert.
  /// - Returns: The `Message` representation of the JSON string.
  /// - Throws: `LiteRTLMError` if the JSON string is invalid.
  public static func jsonToMessage(_ jsonString: String) throws -> Message {
    guard let data = jsonString.data(using: .utf8),
      let jsonObject = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    else {
      throw LiteRTLMError.message(.failedToConvertToJson)
    }

    var contents: [Content] = []
    if let contentArray = jsonObject["content"] as? [[String: Any]] {
      for item in contentArray {
        if let type = item["type"] as? String {
          switch type {
          case "text":
            if let text = item["text"] as? String {
              contents.append(.text(text))
            }
          case "tool_response":
            if let name = item["name"] as? String, let response = item["response"] {
              let id = item["id"] as? String ?? ""
              contents.append(.toolResponse(name: name, response: response, id: id))
            }
          case "image":
            if let path = item["path"] as? String {
              contents.append(.imageFile(path))
            } else if let blob = item["blob"] as? String, let data = Data(base64Encoded: blob) {
              contents.append(.imageData(data))
            }
          case "audio":
            if let path = item["path"] as? String {
              contents.append(.audioFile(path))
            } else if let blob = item["blob"] as? String, let data = Data(base64Encoded: blob) {
              contents.append(.audioData(data))
            }
          default: break
          }
        }
      }
    }

    var channels: [String: String] = [:]
    if let channelsDict = jsonObject["channels"] as? [String: Any] {
      for (key, value) in channelsDict {
        if let strValue = value as? String {
          channels[key] = strValue
        }
      }
    }

    var toolCalls: [ToolCall] = []
    if let toolCallsArray = jsonObject["tool_calls"] as? [[String: Any]] {
      for item in toolCallsArray {
        if let function = item["function"] as? [String: Any],
          let name = function["name"] as? String,
          let args = function["arguments"] as? [String: Any]
        {
          let id = item["id"] as? String ?? ""
          toolCalls.append(ToolCall(name: name, id: id, arguments: args))
        }
      }
    }

    if contents.isEmpty && channels.isEmpty && toolCalls.isEmpty {
      throw LiteRTLMError.message(.invalidContent)
    }

    let roleRaw = jsonObject["role"] as? String ?? "user"
    let role: Role
    switch roleRaw {
    case "assistant", "model":
      role = .model
    default:
      role = Role(rawValue: roleRaw) ?? .user
    }

    return Message(
      contents: Contents(contents: contents),
      role: role,
      channels: channels,
      toolCalls: toolCalls
    )
  }
}
