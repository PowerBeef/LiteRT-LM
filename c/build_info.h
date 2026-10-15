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

// Agello: how this runtime was built (agello/workspace_status.sh).
#ifndef THIRD_PARTY_ODML_LITERT_LM_C_BUILD_INFO_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_BUILD_INFO_H_

#if defined(__APPLE__)
#include "api_export.h"  // NOLINT
#else
#include "c/api_export.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// A JSON object: build_id (12 hex digits, or "dev" outside --config=agello), and the litert_lm,
// litert and ml_drift commits (a commit, "+<hash>" when built with uncommitted changes) and the
// build flags. Static storage: don't free it.
LITERT_LM_C_API_EXPORT
const char* litert_lm_get_build_info(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_BUILD_INFO_H_
