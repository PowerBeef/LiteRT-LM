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

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_TELEMETRY_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_TELEMETRY_H_

#include <stdbool.h>

#if defined(__APPLE__)
#include "api_export.h"  // NOLINT
#else
#include "c/api_export.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Telemetry levels: 0 off, 1 counters and anomalies (the default), 2 spans,
// 3 spans and a trace, 4 diagnostic modes that change timing. The
// AGELLO_TELEMETRY environment variable sets the level at load.
LITERT_LM_C_API_EXPORT
void litert_lm_telemetry_set_level(int level);

LITERT_LM_C_API_EXPORT
int litert_lm_telemetry_get_level(void);

// Everything recorded since the last reset, as JSON: counters, span
// aggregates (count, total, percentiles), anomalies and process memory.
// Free the result with litert_lm_telemetry_free_string.
LITERT_LM_C_API_EXPORT
char* litert_lm_telemetry_snapshot_json(void);

LITERT_LM_C_API_EXPORT
void litert_lm_telemetry_free_string(char* text);

// Zeroes counters, span aggregates, anomalies and trace buffers.
LITERT_LM_C_API_EXPORT
void litert_lm_telemetry_reset(void);

// Writes the span trace (level 3) as Chrome/Perfetto JSON. Returns false on an
// I/O error.
LITERT_LM_C_API_EXPORT
bool litert_lm_telemetry_write_trace(const char* path);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_TELEMETRY_H_
