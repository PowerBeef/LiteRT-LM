"""Shift the rotary-embedding positions of a Gemma 4 .litertlm by a constant.

Edits only the rotary path of every signature in the tf_lite_prefill_decode
section (the attention mask and KV-cache index paths are untouched):

  default  ADD(float_position, SHIFT) right after the CAST(input_pos) that feeds
           the rotary angles (tensor name contains 'maybe_rope/convert_element_type').
  --int    ADD(input_pos, SHIFT) in INT32 right before that CAST, so the float
           position itself is shifted (only the CAST reads the shifted value).

Rotary attention depends only on position differences, so in exact arithmetic
the model's outputs are unchanged (FP32 runs are word-for-word identical).
If the re-serialized graph is larger, the weights move by a 16 KB-aligned delta
and the container header's section offsets are updated.

Usage: python patch_rope.py <in.litertlm> <shift> <out.litertlm> [--int]
"""
import os, shutil, struct, subprocess, sys

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as S
from litert_lm_builder import litertlm_header_schema_py_generated as H

INT_MODE = "--int" in sys.argv  # shift the INT32 position before the CAST
argv = [a for a in sys.argv if a != "--int"]
src, shift, dst = argv[1], float(argv[2]), argv[3]

# 1. Locate the prefill/decode section in the container.
with open(src, "rb") as f:
    head = f.read(32)
    header_end = struct.unpack("<Q", head[24:32])[0]
    f.seek(0)
    header = f.read(header_end)
md = H.LiteRTLMMetaData.GetRootAs(header, 32)
section = None
for i in range(md.SectionMetadata().ObjectsLength()):
    o = md.SectionMetadata().Objects(i)
    for j in range(o.ItemsLength()):
        kv = o.Items(j)
        if kv.Key().decode() == "model_type":
            v = H.StringValue(); v.Init(kv.Value().Bytes, kv.Value().Pos)
            if v.Value().decode() == "tf_lite_prefill_decode":
                section = (o.BeginOffset(), o.EndOffset())
assert section, "no tf_lite_prefill_decode section"
begin, end = section

# 2. Read the graph: everything before the first external weight buffer.
with open(src, "rb") as f:
    f.seek(begin)
    probe = f.read(80 * 2**20)
model = S.Model.GetRootAs(probe, 0)
weights_start = min(model.Buffers(i).Offset() for i in range(model.BuffersLength()) if model.Buffers(i).Offset() > 1)
graph = probe[:weights_start]
print(f"section {begin}..{end}; graph {weights_start:,} bytes; unpacking...", flush=True)
mt = S.ModelT.InitFromObj(S.Model.GetRootAs(graph, 0))

# 3. Insert ADD(shift) after each rotary CAST.
add_code = next(i for i, c in enumerate(mt.operatorCodes)
                if max(c.builtinCode, c.deprecatedBuiltinCode) == S.BuiltinOperator.ADD)
const_buffer = S.BufferT()
const_value = np.int32(shift) if INT_MODE else np.float32(shift)
const_buffer.data = np.frombuffer(const_value.tobytes(), dtype=np.uint8)
mt.buffers.append(const_buffer)
const_buffer_index = len(mt.buffers) - 1

patched = 0
for sig in mt.signatureDefs:
    g = mt.subgraphs[sig.subgraphIndex]
    for op_index, op in enumerate(list(g.operators)):
        out = g.tensors[op.outputs[0]]
        name = out.name.decode() if isinstance(out.name, bytes) else out.name
        if "maybe_rope/convert_element_type" not in name or out.type != S.TensorType.FLOAT32:
            continue
        if INT_MODE:
            # INT32 position -> ADD(shift) -> CAST: only the CAST reads the
            # shifted value (the same INT32 tensor also feeds the mask).
            cast_in = op.inputs[0]
            src_t = g.tensors[cast_in]
            const = S.TensorT()
            const.name = f"rope_position_shift_int_{int(shift)}".encode()
            const.type = S.TensorType.INT32
            const.shape = np.array([1], dtype=np.int32)
            const.buffer = const_buffer_index
            g.tensors.append(const)
            shifted = S.TensorT()
            shifted.name = ((src_t.name.decode() if isinstance(src_t.name, bytes) else src_t.name) + "_shifted").encode()
            shifted.type = S.TensorType.INT32
            shifted.shape = np.array(src_t.shape, dtype=np.int32)
            shifted.buffer = 0
            g.tensors.append(shifted)
            const_index, shifted_index = len(g.tensors) - 2, len(g.tensors) - 1
            add = S.OperatorT()
            add.opcodeIndex = add_code
            add.inputs = np.array([cast_in, const_index], dtype=np.int32)
            add.outputs = np.array([shifted_index], dtype=np.int32)
            add.builtinOptionsType = S.BuiltinOptions.AddOptions
            add.builtinOptions = S.AddOptionsT()
            op.inputs = np.array([shifted_index] + list(op.inputs[1:]), dtype=np.int32)
            g.operators.insert(g.operators.index(op), add)
            patched += 1
            print(f"  {sig.signatureKey.decode()}: INT32 ADD before op {op_index} ({name[-45:]}), shape {list(src_t.shape)}")
            continue
        cast_out = op.outputs[0]
        const = S.TensorT()
        const.name = f"rope_position_shift_{int(shift)}".encode()
        const.type = S.TensorType.FLOAT32
        const.shape = np.array([1], dtype=np.int32)
        const.buffer = const_buffer_index
        g.tensors.append(const)
        shifted = S.TensorT()
        shifted.name = (name + "_shifted").encode()
        shifted.type = S.TensorType.FLOAT32
        shifted.shape = np.array(out.shape, dtype=np.int32)
        shifted.buffer = 0
        g.tensors.append(shifted)
        const_index, shifted_index = len(g.tensors) - 2, len(g.tensors) - 1
        for other in g.operators:
            other.inputs = np.array([shifted_index if t == cast_out else t for t in other.inputs], dtype=np.int32)
        add = S.OperatorT()
        add.opcodeIndex = add_code
        add.inputs = np.array([cast_out, const_index], dtype=np.int32)
        add.outputs = np.array([shifted_index], dtype=np.int32)
        add.builtinOptionsType = S.BuiltinOptions.AddOptions
        add.builtinOptions = S.AddOptionsT()
        g.operators.insert(g.operators.index(op) + 1, add)
        patched += 1
        print(f"  {sig.signatureKey.decode()}: ADD after op {op_index} ({name[-45:]}), shape {list(out.shape)}")
assert patched >= 4, f"expected a rotary CAST in each of the 4 signatures, patched {patched}"

# 4. Re-serialize. If the graph grew, move the weights by a 16 KB-aligned delta
#    (offset fields are fixed-width, so updating them doesn't change the size).
def pack():
    builder = flatbuffers.Builder(len(graph) + 2**20)
    builder.Finish(mt.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())

new_graph = pack()
ALIGN = 16384
delta = 0 if len(new_graph) <= weights_start else -(-(len(new_graph) - weights_start) // ALIGN) * ALIGN
if delta:
    for b in mt.buffers:
        if b.offset > 1:
            b.offset += delta
    size_before = len(new_graph)
    new_graph = pack()
    assert len(new_graph) == size_before, "offset update changed the graph size"
new_weights_start = weights_start + delta
print(f"new graph {len(new_graph):,} bytes; weights move by {delta:,} bytes", flush=True)

# 5. Write: [0, begin) + new graph + padding + weights + everything after the
#    section shifted by delta, then fix the header's section offsets in place.
CHUNK = 64 * 2**20
def copy_range(fin, fout, start, length):
    fin.seek(start)
    while length:
        data = fin.read(min(CHUNK, length))
        fout.write(data)
        length -= len(data)

if not delta:
    subprocess.run(["cp", "-c", src, dst], check=True)
    with open(dst, "r+b") as f:
        f.seek(begin)
        f.write(new_graph + b"\0" * (weights_start - len(new_graph)))
else:
    total = os.path.getsize(src)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        copy_range(fin, fout, 0, begin)
        fout.write(new_graph + b"\0" * (new_weights_start - len(new_graph)))
        copy_range(fin, fout, begin + weights_start, total - (begin + weights_start))
    with open(dst, "r+b") as f:
        header = bytearray(f.read(header_end))
        meta = H.LiteRTLMMetaData.GetRootAs(bytes(header), 32)
        objects = meta.SectionMetadata()
        for i in range(objects.ObjectsLength()):
            o = objects.Objects(i)
            for slot, value in ((6, o.BeginOffset()), (8, o.EndOffset())):
                if value > begin:  # this section's end, and every later section
                    field = o._tab.Pos + o._tab.Offset(slot)
                    struct.pack_into("<Q", header, field, value + delta)
        f.seek(0)
        f.write(header)
print(f"wrote {dst} ({patched} rotary CASTs shifted by {shift:+g})")
