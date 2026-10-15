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

// Agello (T3): the CPU view of a Metal buffer. Apple GPUs share memory with the CPU, so the bytes
// of a buffer with shared storage can be copied as they are, in the GPU's own layout, without a
// lock's layout conversion.
#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_METAL_RAW_BUFFER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_METAL_RAW_BUFFER_H_

#include <cstddef>
#include <optional>

namespace litert::lm {

struct MetalRawBuffer {
  void* contents = nullptr;
  size_t length = 0;
};

// The contents and length of `mtl_buffer` (an id<MTLBuffer>) when its storage is shared;
// nullopt otherwise, and on platforms without Metal.
std::optional<MetalRawBuffer> GetMetalRawBuffer(void* mtl_buffer);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_METAL_RAW_BUFFER_H_
