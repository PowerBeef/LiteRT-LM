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

#include "c/telemetry.h"

#include <cstdlib>
#include <cstring>
#include <string>

#include "agello/agello_telemetry.h"  // from @agello_telemetry

extern "C" {

void litert_lm_telemetry_set_level(int level) {
  agello::telemetry::SetLevel(level);
}

int litert_lm_telemetry_get_level(void) {
  return agello::telemetry::GetLevel();
}

char* litert_lm_telemetry_snapshot_json(void) {
  const std::string json = agello::telemetry::SnapshotJson();
  char* out = static_cast<char*>(std::malloc(json.size() + 1));
  if (out != nullptr) std::memcpy(out, json.c_str(), json.size() + 1);
  return out;
}

void litert_lm_telemetry_free_string(char* text) { std::free(text); }

void litert_lm_telemetry_reset(void) { agello::telemetry::Reset(); }

bool litert_lm_telemetry_write_trace(const char* path) {
  return path != nullptr && agello::telemetry::WriteTrace(path);
}

}  // extern "C"
