// Copyright 2025 The ODML Authors.
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

#include "runtime/components/top_p_cpu_sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "agello/agello_telemetry.h"  // from @agello_telemetry
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/sampling_cpu_util.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/tensor_buffer_util.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

// Computes exact full-vocabulary token log-probability:
// log P(sampled_id) = logits[sampled_id] - logsumexp(logits / temp).
float ComputeTokenLogProb(const float* logits, int vocab_size, int sampled_id,
                          float temperature) {
  if (logits == nullptr || vocab_size <= 0 || sampled_id < 0 ||
      sampled_id >= vocab_size) {
    return -std::numeric_limits<float>::infinity();
  }
  // When temperature <= 0.0 (greedy decoding), evaluate token
  // log-probabilities against the nominal model logits (temperature = 1.0)
  // rather than dividing by epsilon which causes non-argmax logits to
  // underflow to 0.0 and collapse the score to 0.0.
  const float temp = (temperature <= 0.0f) ? 1.0f : temperature;

  float max_logit = logits[0];
  for (int v = 1; v < vocab_size; ++v) {
    if (logits[v] > max_logit) {
      max_logit = logits[v];
    }
  }

  if (!std::isfinite(max_logit)) {
    return (logits[sampled_id] == max_logit)
               ? 0.0f
               : -std::numeric_limits<float>::infinity();
  }

  double sum_exp = 0.0;
  for (int v = 0; v < vocab_size; ++v) {
    sum_exp += std::exp(static_cast<double>((logits[v] - max_logit) / temp));
  }

  if (!std::isfinite(sum_exp) || sum_exp <= 0.0) {
    return 0.0f;
  }

  float log_prob = static_cast<float>(
      (static_cast<double>(logits[sampled_id] - max_logit) / temp) -
      std::log(sum_exp));
  return std::min(0.0f, log_prob);
}

std::vector<float> ComputeExactLogProbsForBatch(
    absl::Span<const float> logits_data_span,
    absl::Span<const int> flat_sampled_ids,
    const std::vector<std::vector<float>>& sampled_scores, int batch_size,
    int sequence_size, float temperature) {
  std::vector<float> scores(batch_size * sequence_size);
  const int total_tokens = batch_size * sequence_size;
  const int vocab_size =
      (total_tokens > 0 && logits_data_span.size() % total_tokens == 0)
          ? logits_data_span.size() / total_tokens
          : 0;
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < sequence_size; ++j) {
      const int idx = i * sequence_size + j;
      const int sampled_id = flat_sampled_ids[idx];
      if (vocab_size > 0 && sampled_id >= 0 && sampled_id < vocab_size) {
        scores[idx] = ComputeTokenLogProb(
            logits_data_span.data() + static_cast<size_t>(idx) * vocab_size,
            vocab_size, sampled_id, temperature);
      } else {
        scores[idx] = std::log(sampled_scores[i][j]);
      }
    }
  }
  return scores;
}

absl::Status ValidateTensor(const TensorBuffer& tensor, int max_num_dims,
                            int batch_size, const std::string& tensor_name) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor.TensorType());
  auto dims = tensor_type.Layout().Dimensions();
  LITERT_ASSIGN_OR_RETURN(int num_significant_dims, NumSignificantDims(tensor));
  if (num_significant_dims > max_num_dims) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The output ", tensor_name, " tensor must have <=", max_num_dims,
        " significant dimension, but got ", num_significant_dims));
  }
  if (dims[0] != batch_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("The output ", tensor_name,
                     " tensor must have the same batch size as the input "
                     "logits tensor, but got ",
                     dims[0], " vs ", batch_size));
  }
  return absl::OkStatus();
}

// Converts an array of float16 values to float32 values.
void ConvertFp16ToFp32(absl::Span<const uint16_t> fp16_values,
                       std::vector<float>& out) {
  out.resize(fp16_values.size());
  for (int i = 0; i < fp16_values.size(); ++i) {
    tflite::half half_val;
    std::memcpy(&half_val, fp16_values.data() + i, sizeof(uint16_t));
    out[i] = static_cast<float>(half_val);
  }
}

// Agello (P6): the first maximum of a row of FP16 logits (IEEE half bits), compared as integers:
// flipping the bits of negatives and the sign bit of positives orders halves like their values.
// Returns -1 when the maximum is a NaN, which the float path handles instead.
int GreedyHalf(const uint16_t* row, int vocab) {
  auto key = [](uint16_t h) -> uint16_t {
    return (h & 0x8000u) ? static_cast<uint16_t>(~h) : static_cast<uint16_t>(h | 0x8000u);
  };
  uint16_t best = 0;
  for (int i = 0; i < vocab; ++i) best = std::max(best, key(row[i]));
  int index = 0;
  while (key(row[index]) != best) ++index;
  const uint16_t h = row[index];
  if ((h & 0x7c00u) == 0x7c00u && (h & 0x3ffu) != 0) return -1;
  // -0 and +0 compare equal as floats: the float path keeps the first of them.
  if ((h & 0x7fffu) == 0) {
    for (int i = 0; i < index; ++i) {
      if ((row[i] & 0x7fffu) == 0) return i;
    }
  }
  return index;
}

}  // namespace

absl::StatusOr<std::unique_ptr<TopPSampler>> TopPSampler::Create(
    int k, float p, float temperature, int batch_size, int sequence_size,
    int seed, bool compute_exact_log_probs) {
  if (k <= 0) {
    return absl::InvalidArgumentError("k must be positive.");
  }
  if (p < 0.0f || p > 1.0f) {
    return absl::InvalidArgumentError("p must be in [0, 1].");
  }
  if (batch_size <= 0) {
    return absl::InvalidArgumentError("batch_size must be positive.");
  }
  if (sequence_size <= 0) {
    return absl::InvalidArgumentError("sequence_size must be positive.");
  }
  if (temperature < 0.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Temperature must be >= 0, but got ", temperature));
  }
  return absl::WrapUnique(new TopPSampler(
      k, p, temperature, batch_size, sequence_size, seed,
      compute_exact_log_probs));
}

absl::Status TopPSampler::SampleToIdAndScoreBuffer(
    const TensorBuffer& logits_tensor, TensorBuffer& ids_tensor,
    TensorBuffer* scores_tensor) {
  auto status = ValidateTensor(logits_tensor, /*max_num_dims=*/3, batch_size_,
                               "input logits");
  if (!status.ok()) {
    return status;
  }
  status =
      ValidateTensor(ids_tensor, /*max_num_dims=*/2, batch_size_, "output ids");
  if (!status.ok()) {
    return status;
  }

  LITERT_ASSIGN_OR_RETURN(auto logits_tensor_type, logits_tensor.TensorType());
  absl::Span<float> logits_data_span;
  if (logits_tensor_type.ElementType() == ElementType::Float32) {
    auto logits_data_or = ReferTensorBufferAsSpan<float>(logits_tensor);
    if (!logits_data_or) {  // Download the data if it is not in host memory.
      LITERT_ASSIGN_OR_RETURN(logits_data_,
                              CopyFromTensorBuffer<float>(logits_tensor));
      logits_data_span = absl::MakeSpan(logits_data_);
    } else {
      logits_data_span = *logits_data_or;
    }
  } else if (logits_tensor_type.ElementType() == ElementType::Float16) {
    LITERT_ASSIGN_OR_RETURN(auto logits_size, logits_tensor.PackedSize());
    logits_f16_.resize(logits_size / sizeof(uint16_t));
    TensorBuffer& mutable_logits_tensor =
        const_cast<TensorBuffer&>(logits_tensor);
    LITERT_RETURN_IF_ERROR(
        mutable_logits_tensor.Read(absl::MakeSpan(logits_f16_)));
    // Agello (P6): greedy sampling (MTP's drafter and verifier, every step)
    // picks the first maximum of the halves as read: no conversion to floats.
    const int rows = batch_size_ * sequence_size_;
    if (k_ == 1 && rows > 0 && logits_f16_.size() % rows == 0 &&
        (scores_tensor == nullptr || !compute_exact_log_probs_)) {
      const int vocab = static_cast<int>(logits_f16_.size() / rows);
      std::vector<int> ids(rows);
      bool greedy = vocab > 0;
      for (int r = 0; r < rows && greedy; ++r) {
        ids[r] = GreedyHalf(logits_f16_.data() + r * vocab, vocab);
        greedy = ids[r] >= 0;
      }
      if (greedy) {
        ::agello::telemetry::ProbeLogitsHalf(logits_f16_.data(), rows, vocab);
        ids_tensor.Write(absl::MakeConstSpan(ids));
        if (scores_tensor != nullptr) {
          status = ValidateTensor(*scores_tensor, /*max_num_dims=*/2,
                                  batch_size_, "output scores");
          if (!status.ok()) {
            return status;
          }
          // A greedy pick's probability is 1, as in TopKTopPSampling.
          std::vector<float> scores(rows, 0.0f);
          scores_tensor->Write(absl::MakeConstSpan(scores));
        }
        return absl::OkStatus();
      }
    }
    ConvertFp16ToFp32(logits_f16_, logits_data_);
    logits_data_span = absl::MakeSpan(logits_data_);
  } else {
    return absl::InvalidArgumentError(
        "Unsupported logits data type for sampler.");
  }

  // Agello telemetry: NaN/Inf, max, top-1 margin and entropy of the logits (sampled calls).
  ::agello::telemetry::ProbeLogits(
      logits_data_span.data(), static_cast<int64_t>(batch_size_) * sequence_size_,
      static_cast<int64_t>(logits_data_span.size()) /
          std::max<int64_t>(1, static_cast<int64_t>(batch_size_) * sequence_size_));

  std::vector<std::vector<float>> sampled_scores;
  auto sampled_ids =
      TopKTopPSampling(logits_data_span, k_, p_, temperature_, generator_,
                       batch_size_, sequence_size_, sampled_scores);
  if (!sampled_ids.ok()) {
    return sampled_ids.status();
  }
  std::vector<int> flat_sampled_ids(batch_size_ * sequence_size_);
  for (int i = 0; i < batch_size_; ++i) {
    for (int j = 0; j < sequence_size_; ++j) {
      flat_sampled_ids[i * sequence_size_ + j] = (*sampled_ids)[i][j];
    }
  }
  ids_tensor.Write(absl::MakeConstSpan(flat_sampled_ids));
  if (scores_tensor != nullptr) {
    status = ValidateTensor(*scores_tensor, /*max_num_dims=*/2, batch_size_,
                            "output scores");
    if (!status.ok()) {
      return status;
    }
    std::vector<float> scores(batch_size_ * sequence_size_);
    if (!compute_exact_log_probs_) {
      for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < sequence_size_; ++j) {
          // The scores are the log of the probability of the sampled token.
          scores[i * sequence_size_ + j] = std::log(sampled_scores[i][j]);
        }
      }
    } else {
      scores = ComputeExactLogProbsForBatch(
          logits_data_span, absl::MakeConstSpan(flat_sampled_ids),
          sampled_scores, batch_size_, sequence_size_, temperature_);
    }
    scores_tensor->Write(absl::MakeConstSpan(scores));
  }
  return absl::OkStatus();
}

absl::Status TopPSampler::UpdateConfig(
    const proto::SamplerParameters& sampler_params, int batch_size,
    std::shared_ptr<std::default_random_engine> rand_gen) {
  k_ = sampler_params.k();
  p_ = sampler_params.p();
  temperature_ = sampler_params.temperature();
  batch_size_ = batch_size;
  compute_exact_log_probs_ = sampler_params.compute_exact_log_probs();
  if (rand_gen != nullptr) {
    generator_ = rand_gen;
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
