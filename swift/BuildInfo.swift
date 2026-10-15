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

/// How the loaded runtime was built (Agello's builds; see `c/build_info.h`).
public enum BuildInfo {
  /// The JSON object: `build_id`, and the `litert_lm`, `litert` and `ml_drift` commits and `flags`.
  public static var json: String { String(cString: litert_lm_get_build_info()) }

  /// `json` decoded.
  public static var fields: [String: String] {
    (try? JSONSerialization.jsonObject(with: Data(json.utf8))) as? [String: String] ?? [:]
  }

  /// 12 hex digits for `--config=agello` builds, `dev` otherwise.
  public static var id: String? { fields["build_id"] }
}
