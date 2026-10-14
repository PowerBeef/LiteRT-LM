// Copyright 2026 Patrice Déry (Agello, PowerBeef/agello-litert-lm)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import CLiteRTLM
import Foundation

/// Probes into the engine (agello/telemetry): counters, span timings, GPU time,
/// anomalies and memory. Level 1 (counters and anomalies) is on by default; the
/// `AGELLO_TELEMETRY` environment variable sets the level when the engine loads.
public enum Telemetry {
  public enum Level: Int32, Sendable {
    /// Nothing recorded.
    case off = 0
    /// Counters and anomalies.
    case counters = 1
    /// Plus span timings (count, total, percentiles) and signposts for Instruments.
    case spans = 2
    /// Plus a per-thread span trace for `writeTrace(to:)`.
    case trace = 3
    /// Plus diagnostic modes that change timing.
    case deep = 4
  }

  public static var level: Level {
    get { Level(rawValue: litert_lm_telemetry_get_level()) ?? .counters }
    set { litert_lm_telemetry_set_level(newValue.rawValue) }
  }

  /// Everything recorded since the last reset, as JSON.
  public static func snapshotJSON() -> String {
    guard let text = litert_lm_telemetry_snapshot_json() else { return "{}" }
    defer { litert_lm_telemetry_free_string(text) }
    return String(cString: text)
  }

  /// `snapshotJSON()` decoded: counters, spans, gauges, anomalies, facts, memory, cpu.
  public static func snapshot() -> [String: Any] {
    let data = Data(snapshotJSON().utf8)
    return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
  }

  public static func reset() {
    litert_lm_telemetry_reset()
  }

  /// Writes the span trace (level 3 or more) as Chrome/Perfetto JSON.
  @discardableResult
  public static func writeTrace(to url: URL) -> Bool {
    litert_lm_telemetry_write_trace(url.path)
  }
}
