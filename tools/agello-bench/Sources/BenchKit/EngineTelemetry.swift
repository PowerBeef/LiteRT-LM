import Darwin
import Foundation

/// The engine's own telemetry (agello/telemetry in our LiteRT-LM builds): counters, span timings,
/// GPU time, anomalies and memory. Its C functions are looked up at run time, so BenchKit still
/// runs against builds that don't have them (Google's), where every call is a no-op.
public enum EngineTelemetry {
  private typealias SetLevel = @convention(c) (Int32) -> Void
  private typealias GetLevel = @convention(c) () -> Int32
  private typealias Snapshot = @convention(c) () -> UnsafeMutablePointer<CChar>?
  private typealias Free = @convention(c) (UnsafeMutablePointer<CChar>?) -> Void
  private typealias Reset = @convention(c) () -> Void
  private typealias WriteTrace = @convention(c) (UnsafePointer<CChar>?) -> Bool

  private static func symbol<T>(_ name: String, as _: T.Type) -> T? {
    guard let handle = dlopen(nil, RTLD_NOW), let address = dlsym(handle, name) else { return nil }
    return unsafeBitCast(address, to: T.self)
  }

  /// Whether the loaded runtime has telemetry.
  public static var isAvailable: Bool {
    symbol("litert_lm_telemetry_snapshot_json", as: Snapshot.self) != nil
  }

  public static var level: Int32 {
    get { symbol("litert_lm_telemetry_get_level", as: GetLevel.self)?() ?? 0 }
    set { symbol("litert_lm_telemetry_set_level", as: SetLevel.self)?(newValue) }
  }

  public static func reset() {
    symbol("litert_lm_telemetry_reset", as: Reset.self)?()
  }

  public static func snapshot() -> TelemetrySnapshot? {
    guard let take = symbol("litert_lm_telemetry_snapshot_json", as: Snapshot.self),
      let free = symbol("litert_lm_telemetry_free_string", as: Free.self),
      let text = take()
    else { return nil }
    defer { free(text) }
    return try? JSONDecoder().decode(TelemetrySnapshot.self, from: Data(String(cString: text).utf8))
  }

  @discardableResult
  public static func writeTrace(to url: URL) -> Bool {
    guard let write = symbol("litert_lm_telemetry_write_trace", as: WriteTrace.self) else {
      return false
    }
    return url.path.withCString { write($0) }
  }
}

/// One telemetry snapshot: everything recorded since the last reset.
public struct TelemetrySnapshot: Codable, Sendable {
  public struct Span: Codable, Sendable {
    public var count: Int
    public var totalMs: Double
    public var meanUs: Double
    public var p50Us: Double
    public var p90Us: Double
    public var p99Us: Double
    public var maxUs: Double

    enum CodingKeys: String, CodingKey {
      case count
      case totalMs = "total_ms"
      case meanUs = "mean_us"
      case p50Us = "p50_us"
      case p90Us = "p90_us"
      case p99Us = "p99_us"
      case maxUs = "max_us"
    }
  }

  public struct Anomaly: Codable, Sendable {
    public var code: String
    public var severity: Int
    public var a: Int64
    public var b: Int64
    public var message: String
    public var tMs: Double

    enum CodingKeys: String, CodingKey {
      case code, severity, a, b, message
      case tMs = "t_ms"
    }
  }

  public var level: Int
  public var counters: [String: Int64]
  public var spans: [String: Span]
  public var anomalies: [Anomaly]
  public var anomaliesTotal: Int64
  public var memory: [String: Double]?
  /// Facts that outlive resets: the sampler path, the GPU family, and so on.
  public var facts: [String: String]?
  /// Process CPU time since the reset: `user_ms`, `system_ms`.
  public var cpu: [String: Double]?

  enum CodingKeys: String, CodingKey {
    case level, counters, spans, anomalies, memory, facts, cpu
    case anomaliesTotal = "anomalies_total"
  }
}
