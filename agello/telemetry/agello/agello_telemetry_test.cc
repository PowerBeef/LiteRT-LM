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

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#include <gtest/gtest.h>
#include "agello/agello_telemetry.h"

namespace agello::telemetry {
namespace {

class TelemetryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    SetLevel(kTrace);
    Reset();
  }
  void TearDown() override { SetLevel(kCounters); }
};

TEST_F(TelemetryTest, CountersRespectTheLevel) {
  AGELLO_COUNT(kTokensDecoded, 3);
  SetLevel(kOff);
  AGELLO_COUNT(kTokensDecoded, 100);
  SetLevel(kCounters);
  AGELLO_COUNT(kTokensDecoded);
  EXPECT_NE(SnapshotJson().find("\"tokens_decoded\":4"), std::string::npos);
}

TEST_F(TelemetryTest, SpansAggregateAcrossThreads) {
  auto work = [] {
    for (int i = 0; i < 1000; ++i) {
      AGELLO_SPAN(kDecodeStep, i);
    }
  };
  std::thread a(work), b(work);
  a.join();
  b.join();
  const std::string json = SnapshotJson();
  EXPECT_NE(json.find("\"decode.step\":{\"count\":2000"), std::string::npos) << json;
}

TEST_F(TelemetryTest, SpansAreOffBelowLevelTwo) {
  SetLevel(kCounters);
  { AGELLO_SPAN(kSample); }
  EXPECT_EQ(SnapshotJson().find("\"sample\""), std::string::npos);
}

TEST_F(TelemetryTest, PercentilesFollowTheDistribution) {
  // 90 short spans of 1 µs and 10 long ones of 1 ms (24 MHz ticks on Apple silicon; use the
  // recorded durations through RecordGpuSeconds, which takes seconds).
  for (int i = 0; i < 90; ++i) RecordGpuSeconds(kGpuDecode, 1.0, 1.0 + 1e-6);
  for (int i = 0; i < 10; ++i) RecordGpuSeconds(kGpuDecode, 1.0, 1.0 + 1e-3);
  const std::string json = SnapshotJson();
  const size_t at = json.find("\"gpu.decode\":{");
  ASSERT_NE(at, std::string::npos) << json;
  double p50 = 0, p99 = 0;
  std::sscanf(json.c_str() + json.find("\"p50_us\":", at), "\"p50_us\":%lf", &p50);
  std::sscanf(json.c_str() + json.find("\"p99_us\":", at), "\"p99_us\":%lf", &p99);
  EXPECT_LT(p50, 2.0);
  EXPECT_GT(p99, 500.0);
}

TEST_F(TelemetryTest, InternedNamesAreStable) {
  const int a = InternSpan("LiteRT::Run[Buffer sync]");
  const int b = InternSpan("LiteRT::Run[Buffer sync]");
  EXPECT_GE(a, kStaticSpanCount);
  EXPECT_EQ(a, b);
  { ScopedSpan span(a); }
  EXPECT_NE(SnapshotJson().find("\"LiteRT::Run[Buffer sync]\""), std::string::npos);
}

TEST_F(TelemetryTest, AnomaliesAreKeptAndCounted) {
  ReportAnomaly(kSlowWait, kWarning, 1500, 7, "session wait");
  const std::string json = SnapshotJson();
  EXPECT_NE(json.find("\"code\":\"slow_wait\""), std::string::npos) << json;
  EXPECT_NE(json.find("\"anomalies\":1"), std::string::npos) << json;
}

TEST_F(TelemetryTest, TraceIsWritten) {
  { AGELLO_SPAN(kPrefillChunk, 1024); }
  const std::string path = ::testing::TempDir() + "/agello_trace.json";
  ASSERT_TRUE(WriteTrace(path));
  std::ifstream file(path);
  std::stringstream contents;
  contents << file.rdbuf();
  EXPECT_NE(contents.str().find("\"name\":\"prefill.chunk\""), std::string::npos);
  EXPECT_NE(contents.str().find("\"arg\":1024"), std::string::npos);
}

}  // namespace
}  // namespace agello::telemetry
