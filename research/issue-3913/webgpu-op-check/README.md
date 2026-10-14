# WebGPU delegate single-op check

Checks the two delegate issues described in the #3913 follow-up as single ops.
All four models were exact on LiteRT-LM v0.18.0's WebGPU accelerator (fully
accelerated) at N=8/F=4, N=128/F=128, N=1024/F=128 and N=2048/F=256, so the
failures seen inside the patched graph came from the surrounding ops.

- `make_models.py <out>` (env `N`, `F`): FULLY_CONNECTED with float32 weights
  and keep_num_dims; REDUCE_MAX over axis 2 of a 4-D tensor, over the last
  axis of a 3-D tensor, and over the last axis of a 4-D tensor (control).
  Needs `ai-edge-litert`, `flatbuffers`, `numpy` (see ../requirements.txt).
- `run_op.c`: runs a model on `cpu` or `gpu` through the LiteRT C API exported
  by `libCLiteRTLM_mac.dylib` (the runner's `.build/artifacts`), with LiteRT's
  `litert/c` headers at the commit pinned by v0.18.0 (`26895c9f`).

```sh
clang -std=c11 -I <litert-headers> run_op.c -L <dylib dir> -lCLiteRTLM_mac \
  -Wl,-rpath,<dylib dir> -o run_op
./run_op model.tflite model.in.bin model.gpu.out.bin gpu
```
