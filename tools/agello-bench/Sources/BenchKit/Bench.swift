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

  public init(
    model: URL, backend: String = "gpu", fp32: Bool = false, maxNumTokens: Int = 8_192,
    mtp: Bool = false, cacheDir: URL? = nil
  ) {
    self.model = model
    self.backend = backend
    self.fp32 = fp32
    self.maxNumTokens = maxNumTokens
    self.mtp = mtp
    self.cacheDir = cacheDir
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
  private let engine: Engine

  /// Load the engine. Engine-level flags are process-wide, so set them here
  /// before anything else creates an engine.
  public static func load(
    _ settings: EngineSettings, label: String, hashModel: Bool = false
  ) async throws -> Bench {
    ExperimentalFlags.optIntoExperimentalAPIs()
    ExperimentalFlags.enableBenchmark = true
    ExperimentalFlags.enableSpeculativeDecoding = settings.mtp

    var cache = "off"
    if let dir = settings.cacheDir {
      try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
      let contents = try FileManager.default.contentsOfDirectory(atPath: dir.path)
      cache = contents.isEmpty ? "cold" : "warm"
    }
    let engine = Engine(
      engineConfig: try EngineConfig(
        modelPath: settings.model.path, backend: settings.backend == "cpu" ? .cpu() : .gpu,
        maxNumTokens: settings.maxNumTokens, cacheDir: settings.cacheDir?.path,
        activationDataType: settings.fp32 ? .float32 : nil))
    let started = ContinuousClock.now
    try await engine.initialize()
    let loadSeconds = (ContinuousClock.now - started).seconds
    return Bench(
      settings: settings, label: label,
      run: "\(label)-\(Int(Date().timeIntervalSince1970))-\(UUID().uuidString.prefix(4))",
      machine: .capture(), loadSeconds: loadSeconds, cache: cache,
      modelSHA256: hashModel ? try Machine.sha256(of: settings.model) : nil, engine: engine)
  }

  public func run(_ item: BenchCase, rep: Int = 0, warmup: Bool = false) async throws -> CaseRecord {
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
    }
    let conversation = try await engine.createConversation(
      with: ConversationConfig(samplerConfig: try SamplerConfig(topK: 1, topP: 1, temperature: 0)))
    let started = ContinuousClock.now
    let reply = try await conversation.sendMessage(
      Message(prompt), maxOutputTokens: maxOutputTokens
    ).toString
    let seconds = (ContinuousClock.now - started).seconds
    let info = try conversation.getBenchmarkInfo()
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
}

extension Duration {
  var seconds: Double {
    let (s, attoseconds) = components
    return Double(s) + Double(attoseconds) / 1e18
  }
}
