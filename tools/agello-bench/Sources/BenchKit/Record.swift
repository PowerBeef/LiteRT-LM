import Foundation

/// One JSONL line: a case's result. The first block of keys matches the #3913
/// research runner's output (`research/issue-3913/summarize.py` reads them).
public struct CaseRecord: Codable, Sendable {
  // The research runner's keys.
  public var id: String
  public var backend: String
  /// "default" (the engine's choice: FP16 on the GPU) or "fp32".
  public var activation: String
  public var model: String
  public var sentences: Int
  public var late: Bool
  public var seed: Int
  /// Prompt tokens prefilled, and tokens decoded.
  public var prefill: Int
  public var decode: Int
  /// Digit copies: whether the reply's digits are exactly the reference.
  public var exact: Bool?
  public var expected: String?
  public var output: String
  public var seconds: Double
  public var mtp: Bool
  public var prefillTPS: Double
  public var decodeTPS: Double
  public var firstMismatch: Int?
  public var firstMismatchPosition: Int?

  // Added by agello-bench.
  public var schema = 1
  public var kind: String
  /// The arm: which build this is (`--label`).
  public var label: String
  /// One process run: its engine load, settings and machine.
  public var run: String
  public var rep: Int
  /// Warm-up passes run first and are left out of comparisons.
  public var warmup: Bool
  /// KV cache capacity (maxNumTokens).
  public var kv: Int
  /// The engine's time to first token, in seconds.
  public var ttft: Double
  public var thermal: String
  public var footprintMB: Double
  public var peakFootprintMB: Double
  /// Engine load time for this run, and whether its cache was cold, warm or off.
  public var loadSeconds: Double
  public var cache: String
  public var modelSHA256: String?
  public var machine: Machine
  /// The engine's telemetry for this case (our builds only): counters, span timings, GPU time,
  /// anomalies and memory, reset when the case starts.
  public var telemetry: TelemetrySnapshot? = nil

  /// One line of JSON, keys sorted.
  public func jsonLine() throws -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    return String(decoding: try encoder.encode(self), as: UTF8.self)
  }
}
