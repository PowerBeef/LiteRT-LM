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

// Agello telemetry for Metal command buffers (Objective-C++ only).
#ifndef AGELLO_TELEMETRY_AGELLO_TELEMETRY_METAL_H_
#define AGELLO_TELEMETRY_AGELLO_TELEMETRY_METAL_H_

#ifdef __OBJC__
#import <Metal/Metal.h>

namespace agello::telemetry {

// Call before committing. Counts the command buffer (level 1+); at level 2+ a completion handler
// adds its GPU time to the model run in flight (SetCurrentRun) and its scheduling latency
// (kernelStartTime to GPUStartTime) to gpu.schedule_latency, and reports command-buffer errors.
void AttachGpuTiming(id<MTLCommandBuffer> command_buffer);

}  // namespace agello::telemetry

#endif  // __OBJC__
#endif  // AGELLO_TELEMETRY_AGELLO_TELEMETRY_METAL_H_
