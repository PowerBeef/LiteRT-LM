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

#import "agello/agello_telemetry_metal.h"

#include "agello/agello_telemetry.h"

namespace agello::telemetry {

void AttachGpuTiming(id<MTLCommandBuffer> command_buffer) {
  const int level = GetLevel();
  if (level < kCounters || command_buffer == nil) return;
  Count(kCommandBuffers);
  const int run = CurrentRun();
  [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
    if (buffer.status == MTLCommandBufferStatusError) {
      ReportAnomaly(kCommandBufferError, kError, static_cast<int64_t>(buffer.error.code), run,
                    "Metal command buffer failed");
    }
    if (GetLevel() < kSpans) return;
    RecordGpuSeconds(run, buffer.GPUStartTime, buffer.GPUEndTime);
    // Submit to start: includes waiting behind earlier command buffers of the same run.
    RecordGpuSeconds(kGpuQueueDelay, buffer.kernelStartTime, buffer.GPUStartTime);
  }];
}

}  // namespace agello::telemetry
