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

// Agello telemetry: low-overhead probes for LiteRT-LM, LiteRT and ML Drift on Apple platforms.
// See docs/agello/TELEMETRY.md on the fork's agello branch.
//
// Every probe first reads one relaxed atomic (the level), so a disabled probe costs a load and a
// branch. Levels:
//   0 off
//   1 counters and anomalies
//   2 + span aggregates (count, total, max, log2 histogram) and DynamicTracing signposts
//   3 + per-thread span ring buffers, written as a Chrome/Perfetto trace
//   4 + diagnostic modes that change timing (per-kernel GPU timing, numerical probes)
// The AGELLO_TELEMETRY environment variable sets the level when the library loads.
// AGELLO_TELEMETRY_OUT=<path> writes the snapshot there at exit (and the trace at level 3+).
#ifndef AGELLO_TELEMETRY_AGELLO_TELEMETRY_H_
#define AGELLO_TELEMETRY_AGELLO_TELEMETRY_H_

#include <atomic>
#include <cstdint>
#include <string>

namespace agello::telemetry {

enum Level : int { kOff = 0, kCounters = 1, kSpans = 2, kTrace = 3, kDeep = 4 };

// X(identifier, "name"): counters. Names appear in snapshots; keep them stable.
#define AGELLO_TELEMETRY_COUNTERS(X)                         \
  X(kTokensPrefilled, "tokens_prefilled")                   \
  X(kTokensDecoded, "tokens_decoded")                       \
  X(kRunsPrefill, "runs_prefill")                           \
  X(kRunsDecode, "runs_decode")                             \
  X(kRunsVerify, "runs_verify")                             \
  X(kRunsDrafter, "runs_drafter")                           \
  X(kRunsOther, "runs_other")                               \
  X(kHostLocks, "host_locks")                               \
  X(kHostLockCopies, "host_lock_copies")                    \
  X(kHostLockCopyBytes, "host_lock_copy_bytes")             \
  X(kCommandBuffers, "command_buffers")                     \
  X(kGpuWaits, "gpu_waits")                                 \
  X(kMaskFills, "mask_fills")                               \
  X(kMaskFillBytes, "mask_fill_bytes")                      \
  X(kTokenHistoryCopies, "token_history_copies")            \
  X(kEmbeddingLookups, "embedding_lookups")                 \
  X(kSamplerGpu, "sampler_gpu")                             \
  X(kSamplerCpu, "sampler_cpu")                             \
  X(kLogitMasks, "logit_masks")                             \
  X(kKvCopies, "kv_copies")                                 \
  X(kKvCopyBytes, "kv_copy_bytes")                          \
  X(kContextSwitches, "context_switches")                   \
  X(kProgramCacheHits, "program_cache_hits")               \
  X(kProgramCacheMisses, "program_cache_misses")           \
  X(kWeightCacheHits, "weight_cache_hits")                 \
  X(kWeightCacheMisses, "weight_cache_misses")             \
  X(kDecodeAllocations, "decode_allocations")              \
  X(kRunOptionParses, "run_option_parses")                 \
  X(kTasksScheduled, "tasks_scheduled")                    \
  X(kMtpRounds, "mtp_rounds")                               \
  X(kMtpDrafted, "mtp_drafted")                             \
  X(kMtpAccepted, "mtp_accepted")                           \
  X(kStopEos, "stop_eos")                                   \
  X(kStopMaxTokens, "stop_max_tokens")                      \
  X(kStopKvFull, "stop_kv_full")                            \
  X(kStopCancelled, "stop_cancelled")                       \
  X(kStreamChunks, "stream_chunks")                         \
  X(kStreamChunkBytes, "stream_chunk_bytes")                \
  X(kErrors, "errors")                                      \
  X(kAnomalies, "anomalies")

enum Counter : int {
#define AGELLO_TELEMETRY_ENUM(id, name) id,
  AGELLO_TELEMETRY_COUNTERS(AGELLO_TELEMETRY_ENUM)
#undef AGELLO_TELEMETRY_ENUM
      kCounterCount
};

// X(identifier, "name"): span kinds with static names. GPU kinds are recorded from command-buffer
// completion handlers (duration only).
#define AGELLO_TELEMETRY_SPANS(X)                     \
  X(kTurnPrefill, "turn.prefill")                     \
  X(kTurnDecode, "turn.decode")                       \
  X(kTemplate, "template")                            \
  X(kTokenize, "tokenize")                            \
  X(kTaskQueueWait, "task.queue_wait")                \
  X(kTaskCallback, "task.callback")                   \
  X(kPrefillChunk, "prefill.chunk")                   \
  X(kEmbeddingLookup, "embedding_lookup")             \
  X(kMaskFill, "mask_fill")                           \
  X(kParamFill, "param_fill")                         \
  X(kEmbeddingLookupPrefill, "prefill.embedding_lookup") \
  X(kMaskFillPrefill, "prefill.mask_fill")            \
  X(kParamFillPrefill, "prefill.param_fill")          \
  X(kBind, "bind")                                    \
  X(kRunPrefill, "run.prefill")                       \
  X(kRunDecode, "run.decode")                         \
  X(kRunVerify, "run.verify")                         \
  X(kRunDrafter, "run.drafter")                       \
  X(kDecodeStep, "decode.step")                       \
  X(kSample, "sample")                                \
  X(kLogitsReadback, "logits_readback")               \
  X(kConstraint, "constraint")                        \
  X(kDetokenize, "detokenize")                        \
  X(kStreamCallback, "stream_callback")               \
  X(kMtpDraft, "mtp.draft")                           \
  X(kMtpVerify, "mtp.verify")                         \
  X(kKvClone, "kv_clone")                             \
  X(kContextSwitch, "context_switch")                 \
  X(kSessionWait, "session_wait")                     \
  X(kEngineInit, "engine.init")                       \
  X(kSessionCreate, "session.create")                 \
  X(kDelegateInvoke, "delegate.invoke")               \
  X(kGpuWaitPrefill, "gpu.wait.prefill")             \
  X(kGpuWaitDecode, "gpu.wait.decode")               \
  X(kGpuWaitOther, "gpu.wait.other")                 \
  X(kGpuPrefill, "gpu.prefill")                       \
  X(kGpuDecode, "gpu.decode")                         \
  X(kGpuVerify, "gpu.verify")                         \
  X(kGpuDrafter, "gpu.drafter")                       \
  X(kGpuOther, "gpu.other")                           \
  X(kGpuQueueDelay, "gpu.queue_delay")

enum Span : int {
#define AGELLO_TELEMETRY_ENUM(id, name) id,
  AGELLO_TELEMETRY_SPANS(AGELLO_TELEMETRY_ENUM)
#undef AGELLO_TELEMETRY_ENUM
      kStaticSpanCount
};
// Spans named at run time (LiteRT's existing trace sites) get ids from kStaticSpanCount up.
inline constexpr int kMaxSpans = kStaticSpanCount + 96;

// X(identifier, "name"): anomaly codes.
#define AGELLO_TELEMETRY_ANOMALIES(X)                             \
  X(kSlowWait, "slow_wait")                                       \
  X(kSessionDestroyedBusy, "session_destroyed_with_pending_tasks") \
  X(kTaskNeverFinished, "task_never_finished")                    \
  X(kInvalidToken, "invalid_sampled_token")                       \
  X(kSamplerFallback, "sampler_fallback_to_cpu")                  \
  X(kLogitMaskFallback, "logit_mask_fallback_to_cpu")             \
  X(kKvContextSwitch, "kv_context_switch")                        \
  X(kConstraintSkipped, "constraint_skipped")                     \
  X(kWeightCacheRebuild, "weight_cache_rebuild")                  \
  X(kProgramCacheRebuild, "program_cache_rebuild")                \
  X(kGpuFamilyUnknown, "gpu_family_unknown")                      \
  X(kAsyncUnavailable, "async_unavailable")                       \
  X(kStopSequenceFailed, "stop_sequence_failed")                  \
  X(kCommandBufferError, "command_buffer_error")                  \
  X(kKvFull, "kv_full")                                           \
  X(kNumerics, "numerics")

enum Anomaly : int {
#define AGELLO_TELEMETRY_ENUM(id, name) id,
  AGELLO_TELEMETRY_ANOMALIES(AGELLO_TELEMETRY_ENUM)
#undef AGELLO_TELEMETRY_ENUM
      kAnomalyCount
};

enum Severity : int { kInfo = 0, kWarning = 1, kError = 2 };

namespace internal {
extern std::atomic<int> level;
extern std::atomic<int64_t> counters[kCounterCount];
uint64_t Now();  // mach_absolute_time ticks (steady clock elsewhere)
void RecordSpan(int span, uint64_t begin, uint64_t end, int64_t arg);
void SignpostBegin(int span, uint64_t id, int64_t arg);
void SignpostEnd(int span, uint64_t id);
}  // namespace internal

inline int GetLevel() { return internal::level.load(std::memory_order_relaxed); }
inline bool Enabled(int at_least) { return GetLevel() >= at_least; }
void SetLevel(int level);

inline void Count(Counter counter, int64_t delta = 1) {
  if (GetLevel() >= kCounters) {
    internal::counters[counter].fetch_add(delta, std::memory_order_relaxed);
  }
}

// Records a span with explicit times (e.g. GPU start/end from a completion handler), in ticks.
inline void RecordDuration(int span, uint64_t begin, uint64_t end, int64_t arg = 0) {
  if (GetLevel() >= kSpans) internal::RecordSpan(span, begin, end, arg);
}

// GPU times arrive in seconds on the mach timebase (MTLCommandBuffer GPUStartTime/GPUEndTime).
void RecordGpuSeconds(int span, double start_seconds, double end_seconds);

// Scoped span: aggregates at level 2+, ring buffer at 3+, DynamicTracing signposts at 2+.
class ScopedSpan {
 public:
  explicit ScopedSpan(int span, int64_t arg = 0) {
    if (span >= 0 && GetLevel() >= kSpans) {
      span_ = span;
      arg_ = arg;
      begin_ = internal::Now();
      internal::SignpostBegin(span_, begin_, arg_);
    }
  }
  ~ScopedSpan() { End(); }
  void SetArg(int64_t arg) { arg_ = arg; }
  void End() {
    if (span_ >= 0) {
      internal::SignpostEnd(span_, begin_);
      internal::RecordSpan(span_, begin_, internal::Now(), arg_);
      span_ = -1;
    }
  }
  ScopedSpan(const ScopedSpan&) = delete;
  ScopedSpan& operator=(const ScopedSpan&) = delete;

 private:
  int span_ = -1;
  int64_t arg_ = 0;
  uint64_t begin_ = 0;
};

inline uint64_t NowTicks() { return internal::Now(); }
double TicksToMs(uint64_t ticks);

// Watchdog for blocking waits: records a session.wait span and reports kSlowWait when the wait
// took `warn_ms` or more (an error from 5x that). `what` must be a string literal. Waits on a
// running generation (synchronous sends) legitimately last as long as the turn.
void CheckSlowWait(uint64_t start_ticks, int64_t id, const char* what, double warn_ms = 60000);

// The GPU wait span for the model run in flight (prefill, decode, or other).
int GpuWaitSpanForCurrentRun();

// Facts about the engine that outlive Reset (sampler path, GPU family, async mode...), shown in
// every snapshot. `key` must be a string literal.
void SetFact(const char* key, const std::string& value);

// Interns a run-time span name (for LiteRT's trace macros); returns a span id or -1 when full.
int InternSpan(const char* name);

// The model run in flight, so GPU completion handlers can attribute their time.
void SetCurrentRun(int gpu_span);
int CurrentRun();

// Anomalies: counted from level 1, kept in a bounded ring, mirrored to os_log.
// `message` must be a string literal (stored by pointer).
void ReportAnomaly(Anomaly code, Severity severity, int64_t a = 0, int64_t b = 0,
                   const char* message = "");

// Snapshot of everything as JSON: counters, span aggregates, anomalies, memory.
std::string SnapshotJson();
// Zeroes counters, span aggregates, anomalies and trace buffers.
void Reset();
// Writes the per-thread span ring buffers as a Chrome/Perfetto trace (level 3). Returns false on
// an I/O error.
bool WriteTrace(const std::string& path);

}  // namespace agello::telemetry

// Convenience macros so call sites compile to nothing extra when the level is low.
#define AGELLO_TELEMETRY_CONCAT_INNER(a, b) a##b
#define AGELLO_TELEMETRY_CONCAT(a, b) AGELLO_TELEMETRY_CONCAT_INNER(a, b)
#define AGELLO_SPAN(kind, ...)                                                       \
  ::agello::telemetry::ScopedSpan AGELLO_TELEMETRY_CONCAT(agello_span_, __LINE__)( \
      ::agello::telemetry::kind, ##__VA_ARGS__)
#define AGELLO_COUNT(counter, ...) \
  ::agello::telemetry::Count(::agello::telemetry::counter, ##__VA_ARGS__)

#endif  // AGELLO_TELEMETRY_AGELLO_TELEMETRY_H_
