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

// Agello telemetry: see agello_telemetry.h.
#include "agello/agello_telemetry.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <os/log.h>
#include <os/signpost.h>
#include <pthread.h>
#include <sys/resource.h>
#if TARGET_OS_IPHONE
#include <os/proc.h>
#endif
#else
#include <chrono>
#endif

namespace agello::telemetry {
namespace internal {

namespace {
int InitialLevel() {
  const char* value = std::getenv("AGELLO_TELEMETRY");
  if (value == nullptr || *value == '\0') return kCounters;
  return std::clamp(std::atoi(value), 0, static_cast<int>(kDeep));
}
}  // namespace

std::atomic<int> level{InitialLevel()};
std::atomic<int64_t> counters[kCounterCount];

uint64_t Now() {
#if defined(__APPLE__)
  return mach_absolute_time();
#else
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
#endif
}

namespace {

// Ticks to nanoseconds.
double TickNs() {
#if defined(__APPLE__)
  static const double ns = [] {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    return static_cast<double>(info.numer) / info.denom;
  }();
  return ns;
#else
  return 1.0;
#endif
}

// Quarter-octave histogram of nanoseconds: index = 4 * floor(log2(ns)) + next two bits.
constexpr int kBuckets = 4 * 48;
int Bucket(uint64_t ns) {
  if (ns < 4) return static_cast<int>(ns);
  const int msb = 63 - __builtin_clzll(ns);
  const int sub = static_cast<int>((ns >> (msb - 2)) & 3);
  return std::min(msb * 4 + sub, kBuckets - 1);
}
double BucketLowerNs(int bucket) {
  if (bucket < 4) return bucket;
  const int msb = bucket / 4, sub = bucket % 4;
  return std::ldexp(4.0 + sub, msb - 2);
}

struct SpanStats {
  std::atomic<int64_t> count{0};
  std::atomic<int64_t> total_ns{0};
  std::atomic<int64_t> max_ns{0};
  std::atomic<int64_t> histogram[kBuckets];
};
SpanStats span_stats[kMaxSpans];

constexpr const char* kStaticSpanNames[] = {
#define AGELLO_TELEMETRY_NAME(id, name) name,
    AGELLO_TELEMETRY_SPANS(AGELLO_TELEMETRY_NAME)
#undef AGELLO_TELEMETRY_NAME
};
constexpr const char* kCounterNames[] = {
#define AGELLO_TELEMETRY_NAME(id, name) name,
    AGELLO_TELEMETRY_COUNTERS(AGELLO_TELEMETRY_NAME)
#undef AGELLO_TELEMETRY_NAME
};
constexpr const char* kAnomalyNames[] = {
#define AGELLO_TELEMETRY_NAME(id, name) name,
    AGELLO_TELEMETRY_ANOMALIES(AGELLO_TELEMETRY_NAME)
#undef AGELLO_TELEMETRY_NAME
};

std::mutex intern_mutex;
std::atomic<const char*> dynamic_names[kMaxSpans - kStaticSpanCount];
std::atomic<int> dynamic_count{0};

const char* SpanName(int span) {
  if (span < kStaticSpanCount) return kStaticSpanNames[span];
  const char* name = dynamic_names[span - kStaticSpanCount].load(std::memory_order_acquire);
  return name != nullptr ? name : "?";
}

// Per-thread ring buffers for the trace (level 3).
struct TraceEvent {
  uint64_t begin;
  uint64_t end;
  int64_t arg;
  int32_t span;
};
constexpr size_t kRingSize = 1 << 16;
struct Ring {
  std::unique_ptr<TraceEvent[]> events{new TraceEvent[kRingSize]};
  std::atomic<uint64_t> written{0};
  uint64_t thread_id = 0;
  std::string thread_name;
};
std::mutex rings_mutex;
std::vector<Ring*> rings;  // Never freed: a few threads, kept for the process lifetime.

Ring* ThreadRing() {
  thread_local Ring* ring = [] {
    auto* created = new Ring();
#if defined(__APPLE__)
    pthread_threadid_np(nullptr, &created->thread_id);
    char name[64] = {};
    pthread_getname_np(pthread_self(), name, sizeof(name));
    created->thread_name = name;
#endif
    std::lock_guard<std::mutex> lock(rings_mutex);
    rings.push_back(created);
    return created;
  }();
  return ring;
}

struct AnomalyRecord {
  uint64_t ticks;
  int code;
  int severity;
  int64_t a;
  int64_t b;
  const char* message;
};
constexpr int kAnomalyRing = 256;
std::mutex anomaly_mutex;
AnomalyRecord anomalies[kAnomalyRing];
int64_t anomaly_total = 0;

std::atomic<int> current_run{kGpuOther};

constexpr const char* kGaugeNames[] = {
#define AGELLO_TELEMETRY_NAME(id, name) name,
    AGELLO_TELEMETRY_GAUGES(AGELLO_TELEMETRY_NAME)
#undef AGELLO_TELEMETRY_NAME
};
constexpr size_t kGaugeReservoir = 4096;
struct GaugeStats {
  int64_t count = 0;
  double sum = 0;
  double min = 0;
  double max = 0;
  std::vector<float> sample;  // Uniform over all values recorded (reservoir sampling).
};
std::mutex gauge_mutex;
GaugeStats gauges[kGaugeCount];
uint64_t gauge_random = 0x9E3779B97F4A7C15ull;

// e^x for x <= 0 (clamped at -87) as 2^(x log2 e), with a degree-6 polynomial for the fraction:
// relative error about 1e-6, and the loop vectorizes (std::exp doesn't).
inline float ExpNonPositive(float x) {
  x = x > -87.0f ? x : -87.0f;
  const float t = x * 1.44269504f;
  const float whole = std::floor(t);
  const float f = t - whole;
  float p = 1.5403530e-4f;
  p = p * f + 1.3333558e-3f;
  p = p * f + 9.6181291e-3f;
  p = p * f + 5.5504109e-2f;
  p = p * f + 2.4022651e-1f;
  p = p * f + 6.9314718e-1f;
  p = p * f + 1.0f;
  const int32_t bits = (static_cast<int32_t>(whole) + 127) << 23;
  float scale;
  std::memcpy(&scale, &bits, sizeof(scale));
  return p * scale;
}

// The value at `fraction` of a sorted sample (nearest rank).
double SamplePercentile(const std::vector<float>& sorted, double fraction) {
  if (sorted.empty()) return 0;
  const size_t rank = static_cast<size_t>(std::ceil(fraction * sorted.size()));
  return sorted[std::min(sorted.size(), std::max<size_t>(rank, 1)) - 1];
}

double CpuMs(bool user) {
  rusage usage = {};
  getrusage(RUSAGE_SELF, &usage);
  const timeval& t = user ? usage.ru_utime : usage.ru_stime;
  return t.tv_sec * 1e3 + t.tv_usec / 1e3;
}
double cpu_base_user = 0;
double cpu_base_system = 0;

#if defined(__APPLE__)
os_log_t SignpostLog() {
  static os_log_t log =
      os_log_create("com.patricedery.agello.litertlm", OS_LOG_CATEGORY_DYNAMIC_TRACING);
  return log;
}
os_log_t EventLog() {
  static os_log_t log = os_log_create("com.patricedery.agello.litertlm", "telemetry");
  return log;
}
#endif

void Append(std::string& out, const char* format, ...) __attribute__((format(printf, 2, 3)));
void Append(std::string& out, const char* format, ...) {
  char buffer[512];
  va_list args;
  va_start(args, format);
  const int n = std::vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  if (n > 0) out.append(buffer, std::min<size_t>(n, sizeof(buffer) - 1));
}

void AppendJsonString(std::string& out, const char* text) {
  out.push_back('"');
  for (const char* p = text; *p != '\0'; ++p) {
    if (*p == '"' || *p == '\\') out.push_back('\\');
    if (static_cast<unsigned char>(*p) < 0x20) continue;
    out.push_back(*p);
  }
  out.push_back('"');
}

double Percentile(const SpanStats& stats, int64_t count, double fraction) {
  const int64_t target = static_cast<int64_t>(std::ceil(fraction * count));
  int64_t seen = 0;
  for (int i = 0; i < kBuckets; ++i) {
    seen += stats.histogram[i].load(std::memory_order_relaxed);
    if (seen >= target) return BucketLowerNs(i);
  }
  return static_cast<double>(stats.max_ns.load(std::memory_order_relaxed));
}

}  // namespace

void RecordSpan(int span, uint64_t begin, uint64_t end, int64_t arg) {
  if (span < 0 || span >= kMaxSpans) return;
  const uint64_t ticks = end > begin ? end - begin : 0;
  const auto ns = static_cast<int64_t>(ticks * TickNs());
  SpanStats& stats = span_stats[span];
  stats.count.fetch_add(1, std::memory_order_relaxed);
  stats.total_ns.fetch_add(ns, std::memory_order_relaxed);
  stats.histogram[Bucket(static_cast<uint64_t>(ns))].fetch_add(1, std::memory_order_relaxed);
  int64_t max = stats.max_ns.load(std::memory_order_relaxed);
  while (ns > max && !stats.max_ns.compare_exchange_weak(max, ns, std::memory_order_relaxed)) {
  }
  if (level.load(std::memory_order_relaxed) >= kTrace) {
    Ring* ring = ThreadRing();
    const uint64_t index = ring->written.fetch_add(1, std::memory_order_relaxed);
    ring->events[index % kRingSize] = {begin, end, arg, span};
  }
}

void SignpostBegin(int span, uint64_t id, int64_t arg) {
#if defined(__APPLE__)
  const os_signpost_id_t signpost = static_cast<os_signpost_id_t>(id | 1);
  switch (span) {
#define AGELLO_TELEMETRY_CASE(kind, name)                                              \
  case kind:                                                                           \
    os_signpost_interval_begin(SignpostLog(), signpost, name, "%lld", (long long)arg); \
    return;
    AGELLO_TELEMETRY_SPANS(AGELLO_TELEMETRY_CASE)
#undef AGELLO_TELEMETRY_CASE
    default:
      os_signpost_interval_begin(SignpostLog(), signpost, "litert", "%{public}s", SpanName(span));
  }
#endif
}

void SignpostEnd(int span, uint64_t id) {
#if defined(__APPLE__)
  const os_signpost_id_t signpost = static_cast<os_signpost_id_t>(id | 1);
  switch (span) {
#define AGELLO_TELEMETRY_CASE(kind, name)                    \
  case kind:                                                 \
    os_signpost_interval_end(SignpostLog(), signpost, name); \
    return;
    AGELLO_TELEMETRY_SPANS(AGELLO_TELEMETRY_CASE)
#undef AGELLO_TELEMETRY_CASE
    default:
      os_signpost_interval_end(SignpostLog(), signpost, "litert");
  }
#endif
}

}  // namespace internal

using namespace internal;  // NOLINT

void SetLevel(int new_level) {
  level.store(std::clamp(new_level, 0, static_cast<int>(kDeep)), std::memory_order_relaxed);
}

void RecordGpuSeconds(int span, double start_seconds, double end_seconds) {
  if (GetLevel() < kSpans || end_seconds <= start_seconds || start_seconds <= 0) return;
  const double ticks_per_second = 1e9 / TickNs();
  RecordSpan(span, static_cast<uint64_t>(start_seconds * ticks_per_second),
             static_cast<uint64_t>(end_seconds * ticks_per_second), 0);
}

void RecordNs(int span, int64_t ns) {
  if (GetLevel() < kSpans || span < 0 || span >= kMaxSpans || ns < 0) return;
  SpanStats& stats = span_stats[span];
  stats.count.fetch_add(1, std::memory_order_relaxed);
  stats.total_ns.fetch_add(ns, std::memory_order_relaxed);
  stats.histogram[Bucket(static_cast<uint64_t>(ns))].fetch_add(1, std::memory_order_relaxed);
  int64_t max = stats.max_ns.load(std::memory_order_relaxed);
  while (ns > max && !stats.max_ns.compare_exchange_weak(max, ns, std::memory_order_relaxed)) {
  }
}

int InternSpan(const char* name) {
  if (name == nullptr) return -1;
  const int known = dynamic_count.load(std::memory_order_acquire);
  for (int i = 0; i < known; ++i) {
    const char* existing = dynamic_names[i].load(std::memory_order_acquire);
    if (existing == name || std::strcmp(existing, name) == 0) return kStaticSpanCount + i;
  }
  std::lock_guard<std::mutex> lock(intern_mutex);
  const int count = dynamic_count.load(std::memory_order_relaxed);
  for (int i = known; i < count; ++i) {
    if (std::strcmp(dynamic_names[i].load(std::memory_order_relaxed), name) == 0) {
      return kStaticSpanCount + i;
    }
  }
  if (count >= kMaxSpans - kStaticSpanCount) return -1;
  dynamic_names[count].store(strdup(name), std::memory_order_release);
  dynamic_count.store(count + 1, std::memory_order_release);
  return kStaticSpanCount + count;
}

double TicksToMs(uint64_t ticks) { return ticks * TickNs() / 1e6; }

void CheckSlowWait(uint64_t start_ticks, int64_t id, const char* what, double warn_ms) {
  const uint64_t end = Now();
  if (GetLevel() >= kSpans) RecordSpan(kSessionWait, start_ticks, end, id);
  const double ms = TicksToMs(end - start_ticks);
  if (ms >= warn_ms) {
    ReportAnomaly(kSlowWait, ms >= 5 * warn_ms ? kError : kWarning, static_cast<int64_t>(ms), id,
                  what);
  }
}

int GpuWaitSpanForCurrentRun() {
  switch (CurrentRun()) {
    case kGpuPrefill:
      return kGpuWaitPrefill;
    case kGpuDecode:
    case kGpuVerify:
    case kGpuDrafter:
      return kGpuWaitDecode;
    default:
      return kGpuWaitOther;
  }
}

namespace {
std::mutex facts_mutex;
std::vector<std::pair<const char*, std::string>> facts;
}  // namespace

void RecordValue(Gauge gauge, double value) {
  if (GetLevel() < kCounters || gauge < 0 || gauge >= kGaugeCount || !std::isfinite(value)) return;
  std::lock_guard<std::mutex> lock(gauge_mutex);
  GaugeStats& stats = gauges[gauge];
  stats.min = stats.count ? std::min(stats.min, value) : value;
  stats.max = stats.count ? std::max(stats.max, value) : value;
  stats.count++;
  stats.sum += value;
  if (stats.sample.size() < kGaugeReservoir) {
    stats.sample.push_back(static_cast<float>(value));
  } else {
    gauge_random = gauge_random * 6364136223846793005ull + 1442695040888963407ull;
    const uint64_t slot = (gauge_random >> 11) % static_cast<uint64_t>(stats.count);
    if (slot < kGaugeReservoir) stats.sample[slot] = static_cast<float>(value);
  }
}

void ProbeLogits(const float* logits, int64_t rows, int64_t vocab) {
  const int level = GetLevel();
  if (level < kCounters || logits == nullptr || rows <= 0 || vocab <= 1) return;
  static std::atomic<uint64_t> calls{0};
  const uint64_t call = calls.fetch_add(1, std::memory_order_relaxed);
  if (level < kDeep) {
    if (call % 32 != 0) return;
    rows = 1;  // Sampled: the first row (position) of the call.
  }
  const int base = CurrentRun() == kGpuDrafter ? kDrafterLogitsMax : kLogitsMax;
  const auto gauge = [base](Gauge model_gauge) {
    return static_cast<Gauge>(base + (model_gauge - kLogitsMax));
  };
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = logits + r * vocab;
    // Pass 1, branchless: non-finite count, max and min of the finite values.
    int64_t nonfinite = 0;
    float top1 = -FLT_MAX, lowest = FLT_MAX;
    for (int64_t begin = 0; begin < vocab; begin += 1024) {
      const int n = static_cast<int>(std::min<int64_t>(1024, vocab - begin));
      const float* block = row + begin;
      int block_nonfinite = 0;
      for (int v = 0; v < n; ++v) {
        const float x = block[v];
        const bool finite = std::fabs(x) <= FLT_MAX;
        block_nonfinite += !finite;
        // Plain max and min over masked values, so the loop vectorizes as a reduction.
        const float high = finite ? x : -FLT_MAX;
        const float low = finite ? x : FLT_MAX;
        top1 = high > top1 ? high : top1;
        lowest = low < lowest ? low : lowest;
      }
      nonfinite += block_nonfinite;
    }
    Count(kLogitsRowsChecked);
    if (nonfinite > 0) {
      Count(kLogitsNonFinite, nonfinite);
      ReportAnomaly(kNumerics, kError, nonfinite, vocab, "NaN or Inf in logits (a of b)");
      if (nonfinite == vocab) continue;
    }
    if (top1 == lowest) {
      ReportAnomaly(kNumerics, kWarning, static_cast<int64_t>(top1), vocab,
                    "flat logits: every token scores the same");
    }
    // Pass 2: the runner-up (or a tie for first) and, from level 2 on finite rows, the entropy at
    // temperature 1: H = log Z - sum(e^x' x') / Z with x' = x - max, summed in float per block.
    const bool entropy = level >= kSpans && nonfinite == 0;
    float top2 = -FLT_MAX;
    int64_t at_top = 0;
    double z = 0, weighted = 0;
    for (int64_t begin = 0; begin < vocab; begin += 1024) {
      const int n = static_cast<int>(std::min<int64_t>(1024, vocab - begin));
      const float* block = row + begin;
      int block_at_top = 0;
      float block_top2 = -FLT_MAX;
      for (int v = 0; v < n; ++v) {
        const float x = block[v];
        block_at_top += x == top1;
        const float below = x < top1 ? x : -FLT_MAX;
        block_top2 = below > block_top2 ? below : block_top2;
      }
      at_top += block_at_top;
      top2 = std::max(top2, block_top2);
      if (entropy) {
        float block_z = 0, block_weighted = 0;
        for (int v = 0; v < n; ++v) {
          const float shifted = block[v] - top1;
          const float e = ExpNonPositive(shifted);
          block_z += e;
          block_weighted += e * shifted;
        }
        z += block_z;
        weighted += block_weighted;
      }
    }
    RecordValue(gauge(kLogitsMax), top1);
    if (at_top > 1) top2 = top1;
    if (top2 > -FLT_MAX) {
      const double margin = static_cast<double>(top1) - top2;
      RecordValue(gauge(kLogitsMargin), margin);
      if (margin < 1.0 / 16 && base == kLogitsMax) Count(kLogitsNearTies);
    }
    if (entropy && z > 0) {
      RecordValue(gauge(kLogitsEntropy), std::log(z) - weighted / z);
      RecordValue(gauge(kLogitsTop1Prob), 1.0 / z);
    }
  }
}

void SetFact(const char* key, const std::string& value) {
  std::lock_guard<std::mutex> lock(facts_mutex);
  for (auto& fact : facts) {
    if (std::strcmp(fact.first, key) == 0) {
      fact.second = value;
      return;
    }
  }
  facts.emplace_back(key, value);
}

void SetCurrentRun(int gpu_span) { current_run.store(gpu_span, std::memory_order_relaxed); }
int CurrentRun() { return current_run.load(std::memory_order_relaxed); }

void ReportAnomaly(Anomaly code, Severity severity, int64_t a, int64_t b, const char* message) {
  if (GetLevel() < kCounters) return;
  counters[kAnomalies].fetch_add(1, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(anomaly_mutex);
    anomalies[anomaly_total % kAnomalyRing] = {Now(), code, severity, a, b, message};
    ++anomaly_total;
  }
#if defined(__APPLE__)
  const char* name = kAnomalyNames[code];
  if (severity >= kError) {
    os_log_error(EventLog(), "anomaly %{public}s a=%lld b=%lld %{public}s", name, (long long)a,
                 (long long)b, message);
  } else {
    os_log(EventLog(), "anomaly %{public}s a=%lld b=%lld %{public}s", name, (long long)a,
           (long long)b, message);
  }
#endif
}

std::string SnapshotJson() {
  std::string out;
  out.reserve(16384);
  Append(out, "{\"schema\":1,\"level\":%d,\"counters\":{", GetLevel());
  for (int i = 0; i < kCounterCount; ++i) {
    Append(out, "%s\"%s\":%lld", i ? "," : "", kCounterNames[i],
           (long long)counters[i].load(std::memory_order_relaxed));
  }
  out += "},\"spans\":{";
  bool first = true;
  const int span_end = kStaticSpanCount + dynamic_count.load(std::memory_order_acquire);
  for (int s = 0; s < span_end; ++s) {
    const SpanStats& stats = span_stats[s];
    const int64_t count = stats.count.load(std::memory_order_relaxed);
    if (count == 0) continue;
    const double total = stats.total_ns.load(std::memory_order_relaxed);
    if (!first) out += ",";
    first = false;
    AppendJsonString(out, SpanName(s));
    Append(out,
           ":{\"count\":%lld,\"total_ms\":%.3f,\"mean_us\":%.2f,\"p50_us\":%.2f,"
           "\"p90_us\":%.2f,\"p99_us\":%.2f,\"max_us\":%.2f}",
           (long long)count, total / 1e6, total / count / 1e3,
           Percentile(stats, count, 0.50) / 1e3, Percentile(stats, count, 0.90) / 1e3,
           Percentile(stats, count, 0.99) / 1e3,
           stats.max_ns.load(std::memory_order_relaxed) / 1e3);
  }
  out += "},\"anomalies\":[";
  {
    std::lock_guard<std::mutex> lock(anomaly_mutex);
    const int64_t begin = std::max<int64_t>(0, anomaly_total - kAnomalyRing);
    for (int64_t i = begin; i < anomaly_total; ++i) {
      const AnomalyRecord& record = anomalies[i % kAnomalyRing];
      Append(out, "%s{\"code\":\"%s\",\"severity\":%d,\"a\":%lld,\"b\":%lld,\"message\":",
             i > begin ? "," : "", kAnomalyNames[record.code], record.severity,
             (long long)record.a, (long long)record.b);
      AppendJsonString(out, record.message);
      Append(out, ",\"t_ms\":%.3f}", record.ticks * TickNs() / 1e6);
    }
    Append(out, "],\"anomalies_total\":%lld", (long long)anomaly_total);
  }
  out += ",\"facts\":{";
  {
    std::lock_guard<std::mutex> lock(facts_mutex);
    for (size_t i = 0; i < facts.size(); ++i) {
      if (i) out += ",";
      AppendJsonString(out, facts[i].first);
      out += ":";
      AppendJsonString(out, facts[i].second.c_str());
    }
  }
  out += "}";
  out += ",\"gauges\":{";
  {
    std::lock_guard<std::mutex> lock(gauge_mutex);
    bool first_gauge = true;
    for (int g = 0; g < kGaugeCount; ++g) {
      const GaugeStats& stats = gauges[g];
      if (stats.count == 0) continue;
      if (!first_gauge) out += ",";
      first_gauge = false;
      std::vector<float> sorted = stats.sample;
      std::sort(sorted.begin(), sorted.end());
      AppendJsonString(out, kGaugeNames[g]);
      Append(out,
             ":{\"count\":%lld,\"mean\":%.4g,\"min\":%.4g,\"p10\":%.4g,\"p50\":%.4g,"
             "\"p90\":%.4g,\"max\":%.4g}",
             (long long)stats.count, stats.sum / stats.count, stats.min,
             SamplePercentile(sorted, 0.10), SamplePercentile(sorted, 0.50),
             SamplePercentile(sorted, 0.90), stats.max);
    }
  }
  out += "}";
#if defined(__APPLE__)
  task_vm_info_data_t info = {};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) ==
      KERN_SUCCESS) {
    Append(out,
           ",\"memory\":{\"footprint_mb\":%.1f,\"peak_footprint_mb\":%.1f,"
           "\"graphics_footprint_mb\":%.1f",
           info.phys_footprint / 1e6, info.ledger_phys_footprint_peak / 1e6,
           info.ledger_tag_graphics_footprint / 1e6);
#if TARGET_OS_IPHONE
    Append(out, ",\"available_mb\":%.1f", os_proc_available_memory() / 1e6);
#endif
    out += "}";
  }
#endif
  // Process CPU time since the last reset: busy-waiting and host work show up here.
  Append(out, ",\"cpu\":{\"user_ms\":%.1f,\"system_ms\":%.1f}", CpuMs(true) - cpu_base_user,
         CpuMs(false) - cpu_base_system);
  out += "}";
  return out;
}

void Reset() {
  cpu_base_user = CpuMs(true);
  cpu_base_system = CpuMs(false);
  for (auto& counter : counters) counter.store(0, std::memory_order_relaxed);
  for (auto& stats : span_stats) {
    stats.count.store(0, std::memory_order_relaxed);
    stats.total_ns.store(0, std::memory_order_relaxed);
    stats.max_ns.store(0, std::memory_order_relaxed);
    for (auto& bucket : stats.histogram) bucket.store(0, std::memory_order_relaxed);
  }
  {
    std::lock_guard<std::mutex> lock(anomaly_mutex);
    anomaly_total = 0;
  }
  {
    std::lock_guard<std::mutex> lock(gauge_mutex);
    for (auto& stats : gauges) stats = GaugeStats();
  }
  std::lock_guard<std::mutex> lock(rings_mutex);
  for (Ring* ring : rings) ring->written.store(0, std::memory_order_relaxed);
}

bool WriteTrace(const std::string& path) {
  FILE* file = std::fopen(path.c_str(), "w");
  if (file == nullptr) return false;
  std::string out = "{\"displayTimeUnit\":\"ns\",\"traceEvents\":[";
  bool first = true;
  std::lock_guard<std::mutex> lock(rings_mutex);
  for (const Ring* ring : rings) {
    if (!first) out += ",";
    first = false;
    Append(out, "{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":1,\"tid\":%llu,\"args\":{\"name\":",
           (unsigned long long)ring->thread_id);
    AppendJsonString(out, ring->thread_name.empty() ? "thread" : ring->thread_name.c_str());
    out += "}}";
    const uint64_t written = ring->written.load(std::memory_order_relaxed);
    const uint64_t begin = written > kRingSize ? written - kRingSize : 0;
    for (uint64_t i = begin; i < written; ++i) {
      const TraceEvent& event = ring->events[i % kRingSize];
      out += ",{\"ph\":\"X\",\"pid\":1,\"name\":";
      AppendJsonString(out, SpanName(event.span));
      Append(out, ",\"tid\":%llu,\"ts\":%.3f,\"dur\":%.3f,\"args\":{\"arg\":%lld}}",
             (unsigned long long)ring->thread_id, event.begin * TickNs() / 1e3,
             (event.end - event.begin) * TickNs() / 1e3, (long long)event.arg);
      if (out.size() > (1 << 20)) {
        std::fwrite(out.data(), 1, out.size(), file);
        out.clear();
      }
    }
  }
  out += "]}";
  const bool ok = std::fwrite(out.data(), 1, out.size(), file) == out.size();
  return std::fclose(file) == 0 && ok;
}

namespace {
// AGELLO_TELEMETRY_OUT=<path>: write the snapshot there when the process exits (and the trace to
// <path>.trace.json at level 3+), so any binary reports without code changes.
struct ExitReport {
  ExitReport() {
    if (std::getenv("AGELLO_TELEMETRY_OUT") != nullptr) std::atexit(&Write);
  }
  static void Write() {
    const char* path = std::getenv("AGELLO_TELEMETRY_OUT");
    if (path == nullptr || *path == '\0') return;
    if (FILE* file = std::fopen(path, "w")) {
      const std::string json = SnapshotJson();
      std::fwrite(json.data(), 1, json.size(), file);
      std::fclose(file);
    }
    if (GetLevel() >= kTrace) WriteTrace(std::string(path) + ".trace.json");
  }
};
const ExitReport exit_report;
}  // namespace

}  // namespace agello::telemetry
