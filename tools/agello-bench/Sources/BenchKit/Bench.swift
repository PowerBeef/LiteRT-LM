import Foundation
import LiteRTLM

/// The engine a run measures. One process loads one engine with these.
public struct EngineSettings: Sendable {
  public var model: URL
  /// "gpu" or "cpu".
  public var backend: String
  /// Force FP32 activations (the GPU's default is FP16).
  public var fp32: Bool
  public var maxNumTokens: Int
  /// Speculative decoding with the model's MTP drafter.
  public var mtp: Bool
  /// The serialization cache (compiled programs, weights); nil turns it off.
  public var cacheDir: URL?
  /// LiteRT-LM's benchmark mode, which BenchmarkInfo needs. It also changes the run: Metal waits
  /// busy-spin instead of sleeping, and prefill waits for completion. Off, timings come from the
  /// engine's telemetry (our builds), as an app without benchmark mode would run.
  public var benchmarkMode: Bool

  public init(
    model: URL, backend: String = "gpu", fp32: Bool = false, maxNumTokens: Int = 8_192,
    mtp: Bool = false, cacheDir: URL? = nil, benchmarkMode: Bool = true
  ) {
    self.model = model
    self.backend = backend
    self.fp32 = fp32
    self.maxNumTokens = maxNumTokens
    self.mtp = mtp
    self.cacheDir = cacheDir
    self.benchmarkMode = benchmarkMode
  }
}

/// One loaded engine that runs cases, a fresh greedy conversation per case.
public struct Bench: Sendable {
  public let settings: EngineSettings
  public let label: String
  public let run: String
  public let machine: Machine
  public let loadSeconds: Double
  /// "cold" (empty cache folder), "warm", or "off".
  public let cache: String
  public let modelSHA256: String?
  /// The engine's telemetry while it loaded (our builds only).
  public let loadTelemetry: TelemetrySnapshot?
  /// Nil when the run only has engine benchmarks, which make their own engine.
  private let engine: Engine?

  /// Load the engine. Engine-level flags are process-wide, so set them here
  /// before anything else creates an engine.
  public static func load(
    _ settings: EngineSettings, label: String, hashModel: Bool = false, loadEngine: Bool = true
  ) async throws -> Bench {
    ExperimentalFlags.optIntoExperimentalAPIs()
    ExperimentalFlags.enableBenchmark = settings.benchmarkMode
    if !settings.benchmarkMode, EngineTelemetry.level < 2 { EngineTelemetry.level = 2 }
    ExperimentalFlags.enableSpeculativeDecoding = settings.mtp

    var cache = "off"
    if let dir = settings.cacheDir {
      try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
      let contents = try FileManager.default.contentsOfDirectory(atPath: dir.path)
      cache = contents.isEmpty ? "cold" : "warm"
    }
    var engine: Engine?
    var loadSeconds = 0.0
    EngineTelemetry.reset()
    if loadEngine {
      let loaded = Engine(
        engineConfig: try EngineConfig(
          modelPath: settings.model.path, backend: settings.backend == "cpu" ? .cpu() : .gpu,
          maxNumTokens: settings.maxNumTokens, cacheDir: settings.cacheDir?.path,
          activationDataType: settings.fp32 ? .float32 : nil))
      let started = ContinuousClock.now
      try await loaded.initialize()
      loadSeconds = (ContinuousClock.now - started).seconds
      engine = loaded
    }
    return Bench(
      settings: settings, label: label,
      run: "\(label)-\(Int(Date().timeIntervalSince1970))-\(UUID().uuidString.prefix(4))",
      machine: .capture(), loadSeconds: loadSeconds, cache: cache,
      modelSHA256: hashModel ? try Machine.sha256(of: settings.model) : nil,
      loadTelemetry: EngineTelemetry.snapshot(), engine: engine)
  }

  public func run(_ item: BenchCase, rep: Int = 0, warmup: Bool = false) async throws -> CaseRecord {
    EngineTelemetry.reset()
    var record = try await measure(item, rep: rep, warmup: warmup)
    record.telemetry = EngineTelemetry.snapshot()
    return record
  }

  func measure(_ item: BenchCase, rep: Int, warmup: Bool) async throws -> CaseRecord {
    if item.kind == .engineBenchmark {
      return try await engineBenchmark(item, rep: rep, warmup: warmup)
    }
    guard let engine else { throw BenchError.noEngine }
    let prompt: String
    let digits: String?
    let maxOutputTokens: Int
    switch item.kind {
    case .digitCopy:
      (prompt, digits) = Prompts.digitCopy(item)
      maxOutputTokens = 80
    case .speed:
      (prompt, digits) = (Prompts.speed(item), nil)
      maxOutputTokens = item.decode ?? 256
    case .engineBenchmark:
      preconditionFailure("handled above")
    }
    let conversation = try await engine.createConversation(
      with: ConversationConfig(samplerConfig: try SamplerConfig(topK: 1, topP: 1, temperature: 0)))
    let started = ContinuousClock.now
    let reply = try await conversation.sendMessage(
      Message(prompt), maxOutputTokens: maxOutputTokens
    ).toString
    let seconds = (ContinuousClock.now - started).seconds
    let info = try Timing(conversation, benchmarkMode: settings.benchmarkMode)
    let mismatch = digits.flatMap { Prompts.firstMismatch(reply: reply, digits: $0) }
    let footprint = Footprint.current()
    return CaseRecord(
      id: item.id, backend: settings.backend, activation: settings.fp32 ? "fp32" : "default",
      model: settings.model.lastPathComponent, sentences: item.sentences, late: item.late,
      seed: item.seed, prefill: info.lastPrefillTokenCount, decode: info.lastDecodeTokenCount,
      exact: digits.map { _ in mismatch == nil }, expected: digits, output: reply, seconds: seconds,
      mtp: settings.mtp, prefillTPS: info.lastPrefillTokensPerSecond,
      decodeTPS: info.lastDecodeTokensPerSecond, firstMismatch: mismatch,
      // The reply starts at the last prompt position.
      firstMismatchPosition: mismatch.map { info.lastPrefillTokenCount - 1 + $0 },
      kind: item.kind.rawValue, label: label, run: run, rep: rep, warmup: warmup,
      kv: settings.maxNumTokens, ttft: info.timeToFirstTokenInSecond,
      thermal: ProcessInfo.processInfo.thermalState.label,
      footprintMB: Double(footprint.bytes) / 1e6, peakFootprintMB: Double(footprint.peakBytes) / 1e6,
      loadSeconds: loadSeconds, cache: cache, modelSHA256: modelSHA256, machine: machine)
  }

  /// `LiteRTLM.benchmark`, uncached, as Agello's in-app benchmark calls it.
  func engineBenchmark(_ item: BenchCase, rep: Int, warmup: Bool) async throws -> CaseRecord {
    let prefill = item.prefill ?? 1_024
    let decode = item.decode ?? 256
    let started = ContinuousClock.now
    let info = try await LiteRTLM.benchmark(
      modelPath: settings.model.path, backend: settings.backend == "cpu" ? .cpu() : .gpu,
      prefillTokens: prefill, decodeTokens: decode, cacheDir: ":nocache")
    let seconds = (ContinuousClock.now - started).seconds
    let footprint = Footprint.current()
    return CaseRecord(
      id: item.id, backend: settings.backend, activation: "default",
      model: settings.model.lastPathComponent, sentences: 0, late: false, seed: 0,
      prefill: info.lastPrefillTokenCount, decode: info.lastDecodeTokenCount, exact: nil,
      expected: nil, output: "", seconds: seconds, mtp: settings.mtp,
      prefillTPS: info.lastPrefillTokensPerSecond, decodeTPS: info.lastDecodeTokensPerSecond,
      firstMismatch: nil, firstMismatchPosition: nil, kind: item.kind.rawValue, label: label,
      run: run, rep: rep, warmup: warmup, kv: prefill + decode + 32,
      ttft: info.timeToFirstTokenInSecond, thermal: ProcessInfo.processInfo.thermalState.label,
      footprintMB: Double(footprint.bytes) / 1e6, peakFootprintMB: Double(footprint.peakBytes) / 1e6,
      loadSeconds: info.initTimeInSecond, cache: "off", modelSHA256: modelSHA256, machine: machine)
  }
}

public enum BenchError: Error, CustomStringConvertible {
  case noEngine

  public var description: String {
    switch self {
    case .noEngine: "No engine loaded for this case (the run only expected engine benchmarks)."
    }
  }
}

/// Prefill and decode figures for one turn: from BenchmarkInfo in benchmark mode, otherwise from
/// the engine's telemetry (turn spans and token counters since the case started).
struct Timing {
  var lastPrefillTokenCount = 0
  var lastDecodeTokenCount = 0
  var lastPrefillTokensPerSecond = 0.0
  var lastDecodeTokensPerSecond = 0.0
  var timeToFirstTokenInSecond = 0.0

  init(_ conversation: Conversation, benchmarkMode: Bool) throws {
    if benchmarkMode {
      let info = try conversation.getBenchmarkInfo()
      lastPrefillTokenCount = info.lastPrefillTokenCount
      lastDecodeTokenCount = info.lastDecodeTokenCount
      lastPrefillTokensPerSecond = info.lastPrefillTokensPerSecond
      lastDecodeTokensPerSecond = info.lastDecodeTokensPerSecond
      timeToFirstTokenInSecond = info.timeToFirstTokenInSecond
      return
    }
    guard let snapshot = EngineTelemetry.snapshot() else { return }
    lastPrefillTokenCount = Int(snapshot.counters["tokens_prefilled"] ?? 0)
    lastDecodeTokenCount = Int(snapshot.counters["tokens_decoded"] ?? 0)
    let prefillMs = snapshot.spans["turn.prefill"]?.totalMs ?? 0
    let decodeMs = snapshot.spans["turn.decode"]?.totalMs ?? 0
    if prefillMs > 0 { lastPrefillTokensPerSecond = Double(lastPrefillTokenCount) / prefillMs * 1000 }
    if decodeMs > 0 { lastDecodeTokensPerSecond = Double(lastDecodeTokenCount) / decodeMs * 1000 }
    let firstStepMs = lastDecodeTokenCount > 0 ? decodeMs / Double(lastDecodeTokenCount) : 0
    timeToFirstTokenInSecond = (prefillMs + firstStepMs) / 1000
  }
}

extension Duration {
  var seconds: Double {
    let (s, attoseconds) = components
    return Double(s) + Double(attoseconds) / 1e18
  }
}
