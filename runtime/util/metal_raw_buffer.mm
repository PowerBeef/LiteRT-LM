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

#import <Metal/Metal.h>

#include "runtime/util/metal_raw_buffer.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cstddef>
#include <optional>

namespace litert::lm {

std::optional<MetalRawBuffer> GetMetalRawBuffer(void* mtl_buffer) {
  if (mtl_buffer == nullptr) return std::nullopt;
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)mtl_buffer;
  if (buffer.storageMode != MTLStorageModeShared || buffer.contents == nullptr) {
    return std::nullopt;
  }
  return MetalRawBuffer{buffer.contents, static_cast<size_t>(buffer.length)};
}

// The mapping starts one page before the returned address; that page holds its total length.
void* AllocateSnapshotMemory(size_t length) {
  const size_t page = static_cast<size_t>(getpagesize());
  const size_t total = page + (length + page - 1) / page * page;
  void* base = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (base == MAP_FAILED) return nullptr;
  *static_cast<size_t*>(base) = total;
  return static_cast<char*>(base) + page;
}

void FreeSnapshotMemory(void* addr) {
  if (addr == nullptr) return;
  char* base = static_cast<char*>(addr) - getpagesize();
  munmap(base, *reinterpret_cast<size_t*>(base));
}

}  // namespace litert::lm
