// Copyright 2026 Patrice Déry
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
// Part of Agello's fork of LiteRT-LM (PowerBeef/agello-litert-lm, branch agello): the state
// behind `Conversation.sendMessageStream`, kept apart from LiteRT-LM's callbacks.

import CLiteRTLM
import Foundation
import OSLog

private let logger = Logger(
  subsystem: "com.google.odml.litertlm.swift",
  category: "ConversationStream"
)

extension NSLock {
  /// Runs `body` holding the lock (Foundation's `withLock` needs iOS 16 and macOS 13).
  func locked<T>(_ body: () throws -> T) rethrows -> T {
    lock()
    defer { unlock() }
    return try body()
  }
}

/// One stream callback's chunk, copied out of LiteRT-LM's memory before anything else runs.
@_spi(Testing) public struct StreamChunk: Sendable, Equatable {
  public var isFinal: Bool
  public var error: String?
  public var text: String?

  public init(isFinal: Bool = false, error: String? = nil, text: String? = nil) {
    self.isFinal = isFinal
    self.error = error
    self.text = text
  }

  /// Reads a native chunk, which is only valid during its callback. A chunk that can't be read
  /// ends the stream like an error.
  init(native chunk: OpaquePointer?) {
    let makeError: (String) -> LiteRTLMError = { .conversation(.invalidResponse($0)) }
    do {
      isFinal = try LiteRTLMError.get(false, "litert_lm_stream_chunk_is_final", makeError) {
        litert_lm_stream_chunk_is_final(chunk, $0)
      }
      // A chunk without an error or text yields a nil result.
      let errorPointer: UnsafePointer<CChar>? = try LiteRTLMError.get(
        nil, "litert_lm_stream_chunk_get_error", makeError
      ) { litert_lm_stream_chunk_get_error(chunk, $0) }
      let textPointer: UnsafePointer<CChar>? = try LiteRTLMError.get(
        nil, "litert_lm_stream_chunk_get_text", makeError
      ) { litert_lm_stream_chunk_get_text(chunk, $0) }
      error = errorPointer.map { String(cString: $0) }
      text = textPointer.map { String(cString: $0) }
    } catch {
      isFinal = true
      self.error = error.localizedDescription
      text = nil
    }
  }
}

/// Tool calls a round asked for, handed from a LiteRT-LM callback to the task that runs them.
struct ToolCallBatch: @unchecked Sendable {
  let calls: [[String: Any]]
}

/// The per-message settings every round of one stream sends with.
struct StreamOptions: @unchecked Sendable {
  var repetitionPenaltyConfig: RepetitionPenaltyConfig?
  var noRepeatNgramConfig: NoRepeatNgramConfig?
  var suppressTokensConfig: SuppressTokensConfig?
  /// Output cap of the first round.
  var maxOutputTokens: Int?
  /// Output cap of each round after a tool response; `maxOutputTokens` when nil.
  var followUpMaxOutputTokens: Int?
  var responseFormat: ResponseFormat?
  /// Tool rounds the stream may run before it ends with `recurringToolCallLimitExceeded`.
  var maxToolRounds: Int
}

/// What a stream needs from its conversation; tests record these instead.
struct StreamEffects: Sendable {
  /// Stops the native work behind the stream. Called off LiteRT-LM's threads.
  var cancelNative: @Sendable () -> Void
  /// Runs a round's tool calls and sends their responses as the next round, under a new
  /// callback ID. Throws when that round can't start.
  var startToolRound: @Sendable (StreamContext, ToolCallBatch) async throws -> Void
}

/// The state of one `sendMessageStream`, shared by every native send it makes (the message,
/// then one per tool round).
///
/// LiteRT-LM can report a send's end more than once (a tool call its parser rejects is reported
/// mid-stream, again when the stream flushes, and again by the final parse), and keeps sending
/// text after reporting an error. So the stream finishes once, here, and every later chunk is
/// dropped. Callbacks find this object through ``StreamTable`` by an integer ID, never through
/// a pointer: a chunk that arrives after the stream ended finds nothing instead of freed memory.
final class StreamContext: @unchecked Sendable {
  let continuation: AsyncThrowingStream<Message, Error>.Continuation
  /// Held weakly: the conversation is deleted where its owner drops it, never inside a
  /// LiteRT-LM callback, where deleting it waits for the very task running the callback.
  weak var conversation: Conversation?
  /// The conversation's ``Conversation/streamOwner``, which outlives the weak reference.
  let owner: UInt64
  let options: StreamOptions
  let effects: StreamEffects

  private let lock = NSLock()
  private var finished = false
  private var cancelled = false
  private var nativeCancelScheduled = false
  private var toolRounds = 0
  private var pendingToolCalls: [[String: Any]] = []

  init(
    continuation: AsyncThrowingStream<Message, Error>.Continuation,
    conversation: Conversation?,
    owner: UInt64,
    options: StreamOptions,
    effects: StreamEffects
  ) {
    self.continuation = continuation
    self.conversation = conversation
    self.owner = owner
    self.options = options
    self.effects = effects
  }

  var isFinished: Bool { lock.locked { finished } }
  var isCancelled: Bool { lock.locked { cancelled } }

  /// Ends the stream once; later calls do nothing. Returns whether this call ended it.
  @discardableResult
  func finish(throwing error: Error? = nil) -> Bool {
    let first = lock.locked {
      guard !finished else { return false }
      finished = true
      return true
    }
    // Outside the lock: finishing runs `onTermination` synchronously.
    guard first else { return false }
    if let error {
      continuation.finish(throwing: error)
    } else {
      continuation.finish()
    }
    return true
  }

  func yield(_ message: Message) {
    guard !isFinished else { return }
    continuation.yield(message)
  }

  /// Marks the stream cancelled. Returns false when it already was, or had finished.
  @discardableResult
  func markCancelled() -> Bool {
    lock.locked {
      guard !cancelled, !finished else { return false }
      cancelled = true
      return true
    }
  }

  /// Stops the native work when the consumer stops iterating before the stream ended (the
  /// iterating task was cancelled); LiteRT-LM would otherwise decode on, for nobody.
  func cancelNativeWorkWhenAbandoned() {
    continuation.onTermination = { [weak self] termination in
      guard case .cancelled = termination, let self, self.markCancelled() else { return }
      self.cancelNativeWork()
    }
  }

  /// Stops the native work once, on a queue LiteRT-LM doesn't own: a callback can run while
  /// LiteRT-LM holds the lock that cancelling takes.
  func cancelNativeWork() {
    let schedule = lock.locked {
      guard !nativeCancelScheduled else { return false }
      nativeCancelScheduled = true
      return true
    }
    guard schedule else { return }
    let cancel = effects.cancelNative
    DispatchQueue.global(qos: .userInitiated).async { cancel() }
  }

  func addToolCalls(_ calls: [[String: Any]]) {
    lock.locked { pendingToolCalls += calls }
  }

  enum NextRound {
    case none
    case limitReached
    case toolCalls(ToolCallBatch)
  }

  /// The tool calls the finished round asked for, counted against the round limit.
  func takeNextRound() -> NextRound {
    lock.locked {
      guard !pendingToolCalls.isEmpty else { return .none }
      guard toolRounds < options.maxToolRounds else { return .limitReached }
      toolRounds += 1
      let calls = pendingToolCalls
      pendingToolCalls = []
      return .toolCalls(ToolCallBatch(calls: calls))
    }
  }
}

/// The streams LiteRT-LM may still call back, by the ID each native send passes as its
/// callback's user data. Never locked across a call into LiteRT-LM.
final class StreamTable: @unchecked Sendable {
  static let shared = StreamTable()

  private let lock = NSLock()
  private var nextID: UInt = 1
  private var nextOwner: UInt64 = 1
  private var streams: [UInt: StreamContext] = [:]

  func makeOwner() -> UInt64 {
    lock.locked {
      defer { nextOwner += 1 }
      return nextOwner
    }
  }

  /// Registers one native send of `context`'s stream; IDs are never reused.
  func register(_ context: StreamContext) -> UInt {
    lock.locked {
      let id = nextID
      nextID += 1
      streams[id] = context
      return id
    }
  }

  func context(for id: UInt) -> StreamContext? {
    lock.locked { streams[id] }
  }

  @discardableResult
  func retire(_ id: UInt) -> StreamContext? {
    lock.locked { streams.removeValue(forKey: id) }
  }

  /// The streams of one conversation still waiting for LiteRT-LM.
  func contexts(owner: UInt64) -> [StreamContext] {
    lock.locked { streams.values.filter { $0.owner == owner } }
  }

  /// Forgets a deleted conversation's sends; returns their streams so they can be ended.
  func retireAll(owner: UInt64) -> [StreamContext] {
    lock.locked {
      let ids = streams.filter { $0.value.owner == owner }.map(\.key)
      return ids.compactMap { streams.removeValue(forKey: $0) }
    }
  }

  var count: Int { lock.locked { streams.count } }
}

/// What one callback does with its chunk, for the native send registered as `id`.
func handleStreamChunk(_ chunk: StreamChunk, id: UInt) {
  // A stream that already ended has no entry: its late chunks go nowhere.
  guard let context = StreamTable.shared.context(for: id) else { return }

  if let error = chunk.error {
    StreamTable.shared.retire(id)
    context.finish(throwing: LiteRTLMError.conversation(.invalidResponse(error)))
    return
  }

  if let text = chunk.text {
    do {
      guard let data = text.data(using: .utf8),
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
      else {
        throw LiteRTLMError.conversation(.invalidJson("Invalid JSON chunk"))
      }
      if let toolCalls = json["tool_calls"] as? [[String: Any]] {
        context.addToolCalls(toolCalls)
      }
      if json["content"] != nil || json["channels"] != nil {
        context.yield(try Conversation.jsonToMessage(text))
      }
    } catch {
      logger.error("Failed to parse response JSON: \(error.localizedDescription)")
      StreamTable.shared.retire(id)
      // LiteRT-LM keeps decoding a stream nobody reads any more: stop it.
      if context.finish(throwing: error) { context.cancelNativeWork() }
      return
    }
  }

  guard chunk.isFinal else { return }
  StreamTable.shared.retire(id)
  if context.isCancelled {
    context.finish(throwing: CancellationError())
    return
  }
  switch context.takeNextRound() {
  case .none:
    context.finish()
  case .limitReached:
    context.finish(
      throwing: LiteRTLMError.conversation(
        .recurringToolCallLimitExceeded(limit: context.options.maxToolRounds)))
  case .toolCalls(let batch):
    // Off LiteRT-LM's thread: tools are app code, and the next round calls back into LiteRT-LM.
    Task {
      do {
        try await context.effects.startToolRound(context, batch)
      } catch {
        context.finish(throwing: error)
      }
    }
  }
}

/// The C callback of every native send: the user data is the send's ``StreamTable`` ID.
func streamCallback(userData: UnsafeMutableRawPointer?, chunk: OpaquePointer?) {
  let id = UInt(bitPattern: userData)
  guard id != 0 else { return }
  handleStreamChunk(StreamChunk(native: chunk), id: id)
}

/// Drives the stream logic without a model: tests deliver chunks as LiteRT-LM would.
@_spi(Testing) public final class StreamTestHarness: @unchecked Sendable {
  public let stream: AsyncThrowingStream<Message, Error>
  private var context: StreamContext!
  private let lock = NSLock()
  private var latestID: UInt = 0
  private var cancelNativeCalls = 0
  private var roundsStarted: [[[String: Any]]] = []

  /// The ID of the native send chunks go to by default (the latest round).
  public var currentID: UInt { lock.locked { latestID } }

  /// - Parameters:
  ///   - maxToolRounds: tool rounds allowed before the stream ends with an error.
  ///   - startsNextRound: whether a tool round registers a new send, as a live one does.
  public init(maxToolRounds: Int = 25, startsNextRound: Bool = true) {
    let (stream, continuation) = AsyncThrowingStream<Message, Error>.makeStream()
    self.stream = stream
    let effects = StreamEffects(
      cancelNative: { [weak self] in self?.recordCancelNative() },
      startToolRound: { [weak self] context, batch in
        self?.recordRound(batch.calls)
        if startsNextRound { self?.registerRound(context) }
      })
    context = StreamContext(
      continuation: continuation, conversation: nil, owner: StreamTable.shared.makeOwner(),
      options: StreamOptions(maxToolRounds: maxToolRounds), effects: effects)
    context.cancelNativeWorkWhenAbandoned()
    let id = StreamTable.shared.register(context)
    lock.locked { latestID = id }
  }

  /// Delivers a chunk to the current send, or to `id`.
  public func deliver(_ chunk: StreamChunk, to id: UInt? = nil) {
    handleStreamChunk(chunk, id: id ?? currentID)
  }

  /// Marks the stream cancelled, as `Conversation.cancel()` does.
  public func cancel() { context.markCancelled() }

  public func isRegistered(_ id: UInt) -> Bool { StreamTable.shared.context(for: id) != nil }
  public var isFinished: Bool { context.isFinished }
  public var cancelNativeCount: Int { lock.locked { cancelNativeCalls } }
  public var toolRoundsStarted: Int { lock.locked { roundsStarted.count } }

  private func recordCancelNative() { lock.locked { cancelNativeCalls += 1 } }
  private func recordRound(_ calls: [[String: Any]]) { lock.locked { roundsStarted.append(calls) } }
  private func registerRound(_ context: StreamContext) {
    let id = StreamTable.shared.register(context)
    lock.locked { latestID = id }
  }
}
