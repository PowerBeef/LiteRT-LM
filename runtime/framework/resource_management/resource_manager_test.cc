// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/framework/resource_management/resource_manager.h"

#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_stats.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/framework/resource_management/context_handler/context_handler.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

// A fake executor that reports executor stats and counts stat resets.
class StatsFakeLlmExecutor : public FakeLlmExecutor {
 public:
  explicit StatsFakeLlmExecutor(int* num_resets)
      : FakeLlmExecutor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                        /*decode_tokens_set=*/{}),
        num_resets_(num_resets) {
    stats_.AccumulatePlainStep(absl::Milliseconds(10));
  }

  std::optional<ExecutorStats> GetExecutorStats() const override {
    return stats_;
  }
  void ResetExecutorStats() override { ++*num_resets_; }

 private:
  ExecutorStats stats_{.module_name = std::string(kLlmModuleName)};
  int* num_resets_;
};

// Tasks::Decode receives the `LockedLlmExecutor` wrapper rather than the
// underlying executor, so the stats APIs must be forwarded through it.
TEST(ResourceManagerTest, AcquiredExecutorForwardsExecutorStats) {
  int num_resets = 0;
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<StatsFakeLlmExecutor>(&num_resets),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmExecutor> executor,
                       resource_manager->AcquireExecutor());

  std::optional<ExecutorStats> stats = executor->GetExecutorStats();
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(stats->module_name, kLlmModuleName);
  EXPECT_EQ(stats->plain_steps(), 1);

  executor->ResetExecutorStats();
  EXPECT_EQ(num_resets, 1);
}

// Agello (T3): a fake executor that counts context snapshots (CloneContext).
class CloneCountingFakeLlmExecutor : public FakeLlmExecutor {
 public:
  explicit CloneCountingFakeLlmExecutor(int* num_clones)
      : FakeLlmExecutor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                        /*decode_tokens_set=*/{}),
        num_clones_(num_clones) {}

  absl::StatusOr<std::unique_ptr<LlmContext>> CloneContext() const override {
    ++*num_clones_;
    return CreateNewContext(std::nullopt, RuntimeConfig());
  }

 private:
  int* num_clones_;
};

// Agello (T3): switching away from a deleted session (a helper conversation)
// doesn't snapshot it; switching away from a live one does.
TEST(ResourceManagerTest, SwitchSkipsTheSnapshotOfADeletedSession) {
  int num_clones = 0;
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<CloneCountingFakeLlmExecutor>(&num_clones),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  const SessionConfig config = SessionConfig::CreateDefault();
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<ContextHandler> main,
                       resource_manager->CreateContextHandler(config));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<ContextHandler> helper,
                       resource_manager->CreateContextHandler(config));
  auto use = [&](std::shared_ptr<ContextHandler> handler) {
    return resource_manager->AcquireExecutorWithContextHandler(std::move(handler))
        .status();
  };

  ASSERT_OK(use(main));
  EXPECT_EQ(num_clones, 0);
  // The main session is alive: it's snapshotted.
  ASSERT_OK(use(helper));
  EXPECT_EQ(num_clones, 1);
  // The helper was deleted: nothing will restore it, so no snapshot.
  helper.reset();
  ASSERT_OK(use(main));
  EXPECT_EQ(num_clones, 1);

  // A live session is snapshotted again.
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<ContextHandler> other,
                       resource_manager->CreateContextHandler(config));
  ASSERT_OK(use(other));
  EXPECT_EQ(num_clones, 2);
}

}  // namespace
}  // namespace litert::lm
