"""Build single-op .tflite models for the WebGPU delegate repro, plus inputs
and NumPy references. Usage: python make_models.py <out_dir>"""
import os
import sys

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as S

N, F = int(os.environ.get("N", 8)), int(os.environ.get("F", 4))
rng = np.random.default_rng(7)


def build(path, in_shape, consts, op_code, op_inputs, out_shape, opts_type, opts):
  """One-op model: tensor 0 = input, consts follow, last tensor = output."""
  m = S.ModelT()
  m.version = 3
  m.description = "webgpu-delegate-repro"
  m.buffers = [S.BufferT()]
  tensors = []

  def tensor(shape, dtype, name, data=None):
    t = S.TensorT()
    t.shape = np.array(shape, dtype=np.int32)
    t.type = dtype
    t.name = name
    if data is not None:
      b = S.BufferT()
      b.data = np.frombuffer(np.ascontiguousarray(data).tobytes(), dtype=np.uint8)
      m.buffers.append(b)
      t.buffer = len(m.buffers) - 1
    else:
      m.buffers.append(S.BufferT())
      t.buffer = len(m.buffers) - 1
    tensors.append(t)
    return len(tensors) - 1

  x = tensor(in_shape, S.TensorType.FLOAT32, "x")
  const_ids = {}
  for name, (arr, dtype) in consts.items():
    const_ids[name] = tensor(arr.shape, dtype, name, arr)
  y = tensor(out_shape, S.TensorType.FLOAT32, "y")

  code = S.OperatorCodeT()
  code.builtinCode = op_code
  code.deprecatedBuiltinCode = min(op_code, 127)
  code.version = 1
  m.operatorCodes = [code]

  op = S.OperatorT()
  op.opcodeIndex = 0
  op.inputs = np.array([x if n == "x" else (-1 if n is None else const_ids[n]) for n in op_inputs],
                       dtype=np.int32)
  op.outputs = np.array([y], dtype=np.int32)
  op.builtinOptionsType = opts_type
  op.builtinOptions = opts

  sg = S.SubGraphT()
  sg.tensors = tensors
  sg.inputs = np.array([x], dtype=np.int32)
  sg.outputs = np.array([y], dtype=np.int32)
  sg.operators = [op]
  sg.name = "main"
  m.subgraphs = [sg]

  sig = S.SignatureDefT()
  ti, to = S.TensorMapT(), S.TensorMapT()
  ti.name, ti.tensorIndex = "x", x
  to.name, to.tensorIndex = "y", y
  sig.inputs, sig.outputs = [ti], [to]
  sig.signatureKey = "serving_default"
  sig.subgraphIndex = 0
  m.signatureDefs = [sig]

  b = flatbuffers.Builder(1024)
  b.Finish(m.Pack(b), file_identifier=b"TFL3")
  with open(path, "wb") as f:
    f.write(b.Output())


def ints(shape, lo=-8, hi=9):
  """Small integers: exact in float16, so any mismatch is a real error."""
  return rng.integers(lo, hi, size=shape).astype(np.float32)


def main(out):
  os.makedirs(out, exist_ok=True)
  cases = {}

  # 1. FULLY_CONNECTED, float32 weights, keep_num_dims: [1,N,16] x [F,16]^T.
  x = ints((1, N, 16))
  w = ints((F, 16), -3, 4)
  bias = np.zeros((F,), np.float32)
  fc = S.FullyConnectedOptionsT()
  fc.fusedActivationFunction = S.ActivationFunctionType.NONE
  fc.keepNumDims = True
  build(f"{out}/fc_float_weights.tflite", x.shape,
        {"w": (w, S.TensorType.FLOAT32), "b": (bias, S.TensorType.FLOAT32)},
        S.BuiltinOperator.FULLY_CONNECTED, ["x", "w", "b"], (1, N, F),
        S.BuiltinOptions.FullyConnectedOptions, fc)
  cases["fc_float_weights"] = (x, (x @ w.T + bias))

  # REDUCE_MAX cases. Values put the maximum away from index 0 of the
  # reduced axis, so "returns the first element" is visible.
  def reduce_case(name, in_shape, axis, keep):
    x = ints(in_shape)
    red = S.ReducerOptionsT()
    red.keepDims = keep
    ref = x.max(axis=axis, keepdims=keep)
    build(f"{out}/{name}.tflite", in_shape,
          {"axis": (np.array([axis], np.int32), S.TensorType.INT32)},
          S.BuiltinOperator.REDUCE_MAX, ["x", "axis"], ref.shape,
          S.BuiltinOptions.ReducerOptions, red)
    cases[name] = (x, ref)

  reduce_case("reduce_max_4d_axis2", (1, N, 16, F), 2, True)   # inner axis
  reduce_case("reduce_max_3d_axis2", (1, N, 16), 2, True)      # last axis, 3-D
  reduce_case("reduce_max_4d_axis3", (1, N, 16, F), 3, True)   # last axis, 4-D (control)

  for name, (x, ref) in cases.items():
    x.astype(np.float32).tofile(f"{out}/{name}.in.bin")
    ref.astype(np.float32).tofile(f"{out}/{name}.ref.bin")
    first = np.take(x, 0, axis=2) if name.startswith("reduce") else None
    if first is not None:
      np.expand_dims(first, 2).astype(np.float32).tofile(f"{out}/{name}.first.bin")
    print(name, "in", x.shape, "out", ref.shape)


if __name__ == "__main__":
  main(sys.argv[1])
