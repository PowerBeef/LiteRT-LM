// Run a single-input, single-output .tflite through LiteRT's CompiledModel
// on the CPU or GPU, and write the output as raw float32.
// Usage: run_op <model.tflite> <input.bin> <output.bin> <cpu|gpu>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"

#define CHECK(expr)                                                   \
  do {                                                                \
    LiteRtStatus s_ = (expr);                                         \
    if (s_ != kLiteRtStatusOk) {                                      \
      fprintf(stderr, "%s failed: status %d\n", #expr, (int)s_);      \
      return 1;                                                       \
    }                                                                 \
  } while (0)

static size_t element_count(const LiteRtRankedTensorType* t) {
  size_t n = 1;
  for (unsigned i = 0; i < t->layout.rank; ++i) n *= (size_t)t->layout.dimensions[i];
  return n;
}

static void* read_file(const char* path, size_t* size) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  *size = (size_t)ftell(f);
  fseek(f, 0, SEEK_SET);
  void* data = malloc(*size);
  fread(data, 1, *size, f);
  fclose(f);
  return data;
}

int main(int argc, char** argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s model.tflite in.bin out.bin cpu|gpu\n", argv[0]);
    return 2;
  }
  bool gpu = strcmp(argv[4], "gpu") == 0;

  LiteRtEnvironment env;
  CHECK(LiteRtCreateEnvironment(0, NULL, &env));
  LiteRtModel model;
  CHECK(LiteRtCreateModelFromFile(env, argv[1], &model));
  LiteRtOptions options;
  CHECK(LiteRtCreateOptions(&options));
  CHECK(LiteRtSetOptionsHardwareAccelerators(
      options, gpu ? kLiteRtHwAcceleratorGpu : kLiteRtHwAcceleratorCpu));
  LiteRtCompiledModel compiled;
  CHECK(LiteRtCreateCompiledModel(env, model, options, &compiled));

  bool fully = false;
  CHECK(LiteRtCompiledModelIsFullyAccelerated(compiled, &fully));

  LiteRtSignature signature;
  CHECK(LiteRtGetModelSignature(model, 0, &signature));
  LiteRtTensor in_tensor, out_tensor;
  CHECK(LiteRtGetSignatureInputTensorByIndex(signature, 0, &in_tensor));
  CHECK(LiteRtGetSignatureOutputTensorByIndex(signature, 0, &out_tensor));
  LiteRtRankedTensorType in_type, out_type;
  CHECK(LiteRtGetRankedTensorType(in_tensor, &in_type));
  CHECK(LiteRtGetRankedTensorType(out_tensor, &out_type));

  size_t in_bytes = element_count(&in_type) * sizeof(float);
  size_t out_bytes = element_count(&out_type) * sizeof(float);
  size_t file_bytes = 0;
  void* input = read_file(argv[2], &file_bytes);
  if (!input || file_bytes != in_bytes) {
    fprintf(stderr, "input size %zu, expected %zu\n", file_bytes, in_bytes);
    return 1;
  }
  // 64-byte aligned host memory, as LiteRT requires.
  void* in_mem = NULL;
  void* out_mem = NULL;
  posix_memalign(&in_mem, 64, in_bytes);
  posix_memalign(&out_mem, 64, out_bytes);
  memcpy(in_mem, input, in_bytes);
  memset(out_mem, 0, out_bytes);

  LiteRtTensorBuffer in_buf, out_buf;
  CHECK(LiteRtCreateTensorBufferFromHostMemory(&in_type, in_mem, in_bytes, NULL, &in_buf));
  CHECK(LiteRtCreateTensorBufferFromHostMemory(&out_type, out_mem, out_bytes, NULL, &out_buf));
  CHECK(LiteRtRunCompiledModel(compiled, 0, 1, &in_buf, 1, &out_buf));

  void* host = NULL;
  CHECK(LiteRtLockTensorBuffer(out_buf, &host, kLiteRtTensorBufferLockModeRead));
  FILE* f = fopen(argv[3], "wb");
  fwrite(host, 1, out_bytes, f);
  fclose(f);
  CHECK(LiteRtUnlockTensorBuffer(out_buf));

  printf("%s accelerator=%s fully_accelerated=%d\n", argv[1], argv[4], fully);

  LiteRtDestroyTensorBuffer(in_buf);
  LiteRtDestroyTensorBuffer(out_buf);
  LiteRtDestroyCompiledModel(compiled);
  LiteRtDestroyOptions(options);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env);
  return 0;
}
