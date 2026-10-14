"""Edit the rotary-position math of a Gemma 4 .litertlm (LiteRT-LM #3913).

Patches every section whose graph computes rotary angles from a float copy of
the INT32 position: tf_lite_prefill_decode (all signatures) and
tf_lite_mtp_drafter. The attention mask and KV-cache index paths are untouched.

Modes:
  <shift>          ADD(float_position, SHIFT) right after the rotary CAST.
  <shift> --int    ADD(input_pos, SHIFT) in INT32 right before the rotary CAST.
  --split          Exact rotary phases for positions 0..32767 with float16
                   activations. position = 2048*h + l:
                   - l (0..2047, exact in float16) goes through the existing
                     MUL / SIN / COS;
                   - the phase of 2048*h comes from small precomputed tables,
                     selected with onehot(h);
                   - angle addition combines them.
                   Every new tensor is 4-D and every reduction runs over the
                   last axis, the layout the model's own REDUCE_MAX ops use
                   (other axes, and FULLY_CONNECTED with float weights,
                   computed wrong results on the WebGPU delegate).

In exact arithmetic the model's outputs are unchanged. Sections that grow move
by 16 KB-aligned deltas, and the container header's offsets are updated.

Usage:
  python patch_rope.py <in.litertlm> <shift> <out.litertlm> [--int]
  python patch_rope.py <in.litertlm> --split <out.litertlm>
"""
import os, struct, sys

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as S
from litert_lm_builder import litertlm_header_schema_py_generated as H

INT_MODE = "--int" in sys.argv
args = [a for a in sys.argv[1:] if a != "--int"]
SPLIT = args[1] == "--split"
src, dst = args[0], args[2]
shift = None if SPLIT else float(args[1])

BLOCK = 2048          # float16 represents every integer up to 2048
BLOCKS = 16           # 16 * 2048 = 32768 positions
ALIGN = 16384         # section alignment in the container
TARGETS = ("tf_lite_prefill_decode", "tf_lite_mtp_drafter")


def _aligned_buffer_pack(self, builder):
    """BufferT.Pack honoring the schema's force_align: 16 for inline weights."""
    data = None
    if self.data is not None and len(self.data):
        array = np.ascontiguousarray(np.asarray(self.data, dtype=np.uint8))
        builder.StartVector(1, array.size, 16)
        builder.head = builder.head - array.size
        builder.Bytes[builder.head:builder.head + array.size] = array.tobytes()
        data = builder.EndVector()
    S.BufferStart(builder)
    if data is not None:
        S.BufferAddData(builder, data)
    S.BufferAddOffset(builder, self.offset)
    S.BufferAddSize(builder, self.size)
    return S.BufferEnd(builder)


S.BufferT.Pack = _aligned_buffer_pack


def read_header(path):
    with open(path, "rb") as f:
        header_end = struct.unpack("<Q", f.read(32)[24:32])[0]
        f.seek(0)
        header = f.read(header_end)
    md = H.LiteRTLMMetaData.GetRootAs(header, 32)
    sections = []
    for i in range(md.SectionMetadata().ObjectsLength()):
        o = md.SectionMetadata().Objects(i)
        kind = None
        for j in range(o.ItemsLength()):
            kv = o.Items(j)
            if kv.Key().decode() == "model_type":
                v = H.StringValue(); v.Init(kv.Value().Bytes, kv.Value().Pos)
                kind = v.Value().decode()
        sections.append((o.BeginOffset(), o.EndOffset(), kind))
    return header_end, sections


def name_of(t):
    return t.name.decode() if isinstance(t.name, bytes) else t.name


def patch_section(begin, end, kind):
    """Returns the section's new bytes (length grows by a multiple of ALIGN)."""
    with open(src, "rb") as f:
        f.seek(begin)
        data = f.read(end - begin)
    model = S.Model.GetRootAs(data, 0)
    external = [model.Buffers(i).Offset() for i in range(model.BuffersLength()) if model.Buffers(i).Offset() > 1]
    graph_end = min(external) if external else len(data)
    print(f"{kind}: {len(data):,} bytes, graph {graph_end:,} bytes; unpacking...", flush=True)
    mt = S.ModelT.InitFromObj(S.Model.GetRootAs(data[:graph_end] if external else data, 0))

    def opcode(name):
        code = getattr(S.BuiltinOperator, name)
        return next(i for i, c in enumerate(mt.operatorCodes) if max(c.builtinCode, c.deprecatedBuiltinCode) == code)

    def code_of(o):
        c = mt.operatorCodes[o.opcodeIndex]
        return max(c.builtinCode, c.deprecatedBuiltinCode)

    def add_buffer(array):
        b = S.BufferT()
        b.data = np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.uint8)
        mt.buffers.append(b)
        return len(mt.buffers) - 1

    def buffer_values(tensor):
        b = mt.buffers[tensor.buffer]
        dtype = {S.TensorType.FLOAT32: np.float32, S.TensorType.INT32: np.int32}[tensor.type]
        if b.data is not None and len(b.data):
            return np.frombuffer(bytes(b.data), dtype=dtype)
        return np.frombuffer(data[b.offset:b.offset + b.size], dtype=dtype)

    patched = 0
    for sig in mt.signatureDefs:
        g = mt.subgraphs[sig.subgraphIndex]
        sig_name = sig.signatureKey.decode()

        def tensor(name, ttype, shape, buffer=0):
            t = S.TensorT()
            t.name, t.type, t.buffer = name.encode(), ttype, buffer
            t.shape = np.array(shape, dtype=np.int32)
            g.tensors.append(t)
            return len(g.tensors) - 1

        def const(name, array, ttype):
            return tensor(name, ttype, list(np.shape(array)) or [1], add_buffer(array))

        def op(code, inputs, outputs, options_type=0, options=None):
            o = S.OperatorT()
            o.opcodeIndex = opcode(code)
            o.inputs = np.array(inputs, dtype=np.int32)
            o.outputs = np.array(outputs, dtype=np.int32)
            o.builtinOptionsType, o.builtinOptions = options_type, options
            return o

        def consumers(t):
            return [o for o in g.operators if t in list(o.inputs)]

        def rewire(old, new, skip=()):
            for o in g.operators:
                if not any(o is s for s in skip):
                    o.inputs = np.array([new if t == old else t for t in o.inputs], dtype=np.int32)

        def add_opts():
            o = S.AddOptionsT(); o.fusedActivationFunction = S.ActivationFunctionType.NONE
            return S.BuiltinOptions.AddOptions, o

        def sub_opts():
            o = S.SubOptionsT(); o.fusedActivationFunction = S.ActivationFunctionType.NONE
            return S.BuiltinOptions.SubOptions, o

        def mul_opts():
            return S.BuiltinOptions.MulOptions, S.MulOptionsT()

        def cast_opts(a, b):
            o = S.CastOptionsT(); o.inDataType, o.outDataType = a, b
            return S.BuiltinOptions.CastOptions, o

        def reshape_opts(shape):
            o = S.ReshapeOptionsT(); o.newShape = list(shape)
            return S.BuiltinOptions.ReshapeOptions, o

        def reducer_opts():
            o = S.ReducerOptionsT(); o.keepDims = True
            return S.BuiltinOptions.ReducerOptions, o

        casts = [o for o in g.operators if "maybe_rope/convert_element_type" in name_of(g.tensors[o.outputs[0]])
                 and g.tensors[o.outputs[0]].type == S.TensorType.FLOAT32]
        if not casts:
            continue
        assert len(casts) == 1, f"{kind}/{sig_name}: expected one rotary CAST, found {len(casts)}"
        cast = casts[0]
        cast_in, cast_out = int(cast.inputs[0]), int(cast.outputs[0])
        pos_shape = list(g.tensors[cast_in].shape)

        if not SPLIT and INT_MODE:
            c = const(f"rope_shift_int_{int(shift)}", np.array([int(shift)], np.int32), S.TensorType.INT32)
            shifted = tensor(name_of(g.tensors[cast_in]) + "_shifted", S.TensorType.INT32, pos_shape)
            cast.inputs = np.array([shifted] + list(cast.inputs[1:]), dtype=np.int32)
            g.operators.insert(g.operators.index(cast), op("ADD", [cast_in, c], [shifted], *add_opts()))
            patched += 1
            continue
        if not SPLIT:
            c = const(f"rope_shift_{int(shift)}", np.array([shift], np.float32), S.TensorType.FLOAT32)
            shifted = tensor(name_of(g.tensors[cast_out]) + "_shifted", S.TensorType.FLOAT32, list(g.tensors[cast_out].shape))
            add = op("ADD", [cast_out, c], [shifted], *add_opts())
            rewire(cast_out, shifted)
            g.operators.insert(g.operators.index(cast) + 1, add)
            patched += 1
            continue

        # ---- split: position = 2048*h + l (4-D tensors, last-axis reductions)
        # The MULs read the float position directly (prefill/decode) or after a
        # RESHAPE (drafter: [1] -> [1,1,1]); that whole chain is replaced.
        chain, pos_float = [cast], cast_out
        after = consumers(pos_float)
        if len(after) == 1 and code_of(after[0]) == S.BuiltinOperator.RESHAPE:
            chain.append(after[0])
            pos_float = int(after[0].outputs[0])
        pos_shape = list(g.tensors[pos_float].shape)     # [1, N, 1]
        N = pos_shape[1]
        cand_shape = [1, N, 1, BLOCKS]
        offsets = const("rope_block_offsets", (-BLOCK * np.arange(BLOCKS)).astype(np.int32).reshape(1, 1, 1, BLOCKS), S.TensorType.INT32)
        zero_i = const("rope_zero_i", np.array([0], np.int32), S.TensorType.INT32)
        block_i = const("rope_block_i", np.array([BLOCK], np.int32), S.TensorType.INT32)
        minus1 = const("rope_minus_one", np.array([-1.0], np.float32), S.TensorType.FLOAT32)
        axis = const("rope_axis_last", np.array([3], np.int32), S.TensorType.INT32)
        p4_shape = const("rope_pos4_shape", np.array([1, N, 1, 1], np.int32), S.TensorType.INT32)
        p3_shape = const("rope_pos3_shape", np.array(pos_shape, np.int32), S.TensorType.INT32)
        pos4 = tensor("rope_position4", S.TensorType.INT32, [1, N, 1, 1])
        cand = tensor("rope_candidates", S.TensorType.INT32, cand_shape)
        ge = tensor("rope_candidates_ge0", S.TensorType.BOOL, cand_shape)
        lt = tensor("rope_candidates_lt2048", S.TensorType.BOOL, cand_shape)
        valid = tensor("rope_block_valid", S.TensorType.BOOL, cand_shape)
        onehot = tensor("rope_block_onehot", S.TensorType.FLOAT32, cand_shape)
        cand_f = tensor("rope_candidates_f", S.TensorType.FLOAT32, cand_shape)
        masked = tensor("rope_candidates_masked", S.TensorType.FLOAT32, cand_shape)
        low4 = tensor("rope_position_low4", S.TensorType.FLOAT32, [1, N, 1, 1])
        low = tensor("rope_position_low", S.TensorType.FLOAT32, pos_shape)
        low_ops = [
            op("RESHAPE", [cast_in, p4_shape], [pos4], *reshape_opts([1, N, 1, 1])),
            op("ADD", [pos4, offsets], [cand], *add_opts()),
            op("GREATER_EQUAL", [cand, zero_i], [ge]),
            op("LESS", [cand, block_i], [lt]),
            op("LOGICAL_AND", [ge, lt], [valid]),
            op("CAST", [valid], [onehot], *cast_opts(S.TensorType.BOOL, S.TensorType.FLOAT32)),
            op("CAST", [cand], [cand_f], *cast_opts(S.TensorType.INT32, S.TensorType.FLOAT32)),
            op("SELECT_V2", [valid, cand_f, minus1], [masked]),
            op("REDUCE_MAX", [masked, axis], [low4], *reducer_opts()),
            op("RESHAPE", [low4, p3_shape], [low], *reshape_opts(pos_shape)),
        ]
        muls = consumers(pos_float)
        assert len(muls) == 2, f"{kind}/{sig_name}: expected 2 rotary MULs, found {len(muls)}"
        rewire(pos_float, low)
        at = g.operators.index(cast)
        for dead in chain:                               # the original CAST (+ RESHAPE) is gone
            g.operators.remove(dead)
        g.operators[at:at] = low_ops

        for mul in muls:
            inv = next(int(t) for t in mul.inputs if int(t) != low)
            freqs = buffer_values(g.tensors[inv]).astype(np.float64).reshape(-1)
            F = freqs.size
            reshape = consumers(int(mul.outputs[0]))[0]
            trig = consumers(int(reshape.outputs[0]))
            sin_op = next(o for o in trig if code_of(o) == S.BuiltinOperator.SIN)
            cos_op = next(o for o in trig if code_of(o) == S.BuiltinOperator.COS)
            sin_l, cos_l = int(sin_op.outputs[0]), int(cos_op.outputs[0])
            out4 = list(g.tensors[sin_l].shape)          # [1, N, 1, F]
            # onehot[1,N,1,16] * (table + 2)[F,16] -> REDUCE_MAX(last) -> [1,N,F,1] -> [1,N,1,F] -> -2
            angles = np.outer(freqs, BLOCK * np.arange(BLOCKS))   # [F, 16]
            t_sin = const(f"rope_table_sin2_{F}", (np.sin(angles) + 2).astype(np.float32), S.TensorType.FLOAT32)
            t_cos = const(f"rope_table_cos2_{F}", (np.cos(angles) + 2).astype(np.float32), S.TensorType.FLOAT32)
            shape_c = const(f"rope_shape_{F}", np.array(out4, np.int32), S.TensorType.INT32)
            two = const(f"rope_minus_two_{F}", np.array([-2.0], np.float32), S.TensorType.FLOAT32)
            sel_shape, red_shape = [1, N, F, BLOCKS], [1, N, F, 1]
            ms, mc = (tensor(f"rope_sel_{n}_{F}", S.TensorType.FLOAT32, sel_shape) for n in ("sin", "cos"))
            rs, rc = (tensor(f"rope_max_{n}_{F}", S.TensorType.FLOAT32, red_shape) for n in ("sin", "cos"))
            rs4, rc4 = (tensor(f"rope_max4_{n}_{F}", S.TensorType.FLOAT32, out4) for n in ("sin", "cos"))
            ts, tc = (tensor(f"rope_high_{n}_{F}", S.TensorType.FLOAT32, out4) for n in ("sin", "cos"))
            p1, p2, p3, p4 = (tensor(f"rope_{F}_p{k}", S.TensorType.FLOAT32, out4) for k in range(4))
            sin_t = tensor(f"rope_{F}_sin", S.TensorType.FLOAT32, out4)
            cos_t = tensor(f"rope_{F}_cos", S.TensorType.FLOAT32, out4)
            new = [
                op("MUL", [onehot, t_sin], [ms], *mul_opts()),
                op("MUL", [onehot, t_cos], [mc], *mul_opts()),
                op("REDUCE_MAX", [ms, axis], [rs], *reducer_opts()),
                op("REDUCE_MAX", [mc, axis], [rc], *reducer_opts()),
                op("RESHAPE", [rs, shape_c], [rs4], *reshape_opts(out4)),
                op("RESHAPE", [rc, shape_c], [rc4], *reshape_opts(out4)),
                op("ADD", [rs4, two], [ts], *add_opts()),
                op("ADD", [rc4, two], [tc], *add_opts()),
                op("MUL", [ts, cos_l], [p1], *mul_opts()),
                op("MUL", [tc, sin_l], [p2], *mul_opts()),
                op("ADD", [p1, p2], [sin_t], *add_opts()),
                op("MUL", [tc, cos_l], [p3], *mul_opts()),
                op("MUL", [ts, sin_l], [p4], *mul_opts()),
                op("SUB", [p3, p4], [cos_t], *sub_opts()),
            ]
            rewire(sin_l, sin_t, skip=new)
            rewire(cos_l, cos_t, skip=new)
            at = max(g.operators.index(sin_op), g.operators.index(cos_op)) + 1
            g.operators[at:at] = new
            patched += 1
            print(f"  {sig_name}: rotary F={F} ({len(consumers(sin_t))}+{len(consumers(cos_t))} consumers rewired)")

    assert patched, f"{kind}: no rotary path found"

    def pack():
        builder = flatbuffers.Builder(graph_end + 2**20)
        builder.Finish(mt.Pack(builder), file_identifier=b"TFL3")
        return bytes(builder.Output())

    new_graph = pack()
    if external:
        delta = 0 if len(new_graph) <= graph_end else -(-(len(new_graph) - graph_end) // ALIGN) * ALIGN
        if delta:
            for b in mt.buffers:
                if b.offset > 1:
                    b.offset += delta
            size = len(new_graph)
            new_graph = pack()
            assert len(new_graph) == size, "offset update changed the graph size"
        out = new_graph + b"\0" * (graph_end + delta - len(new_graph)) + data[graph_end:]
    else:
        delta = 0 if len(new_graph) <= len(data) else -(-(len(new_graph) - len(data)) // ALIGN) * ALIGN
        out = new_graph + b"\0" * (len(data) + delta - len(new_graph))
    print(f"  {patched} rotary paths; section grows by {delta:,} bytes", flush=True)
    return out


# ---------------------------------------------------------------- assemble
header_end, sections = read_header(src)
new_sections = {i: patch_section(b, e, k) for i, (b, e, k) in enumerate(sections) if k in TARGETS}
total = os.path.getsize(src)
shifts, running = [], 0                          # shift of each section's begin
for i, (b, e, k) in enumerate(sections):
    shifts.append(running)
    if i in new_sections:
        running += len(new_sections[i]) - (e - b)

CHUNK = 64 * 2**20
with open(src, "rb") as fin, open(dst, "wb") as fout:
    def copy(start, stop):
        fin.seek(start)
        while start < stop:
            chunk = fin.read(min(CHUNK, stop - start))
            fout.write(chunk)
            start += len(chunk)
    cursor = 0
    for i, (b, e, k) in enumerate(sections):
        copy(cursor, b)                          # header / padding before the section
        if i in new_sections:
            fout.write(new_sections[i])
        else:
            copy(b, e)
        cursor = e
    copy(cursor, total)

with open(dst, "r+b") as f:
    header = bytearray(f.read(header_end))
    objects = H.LiteRTLMMetaData.GetRootAs(bytes(header), 32).SectionMetadata()
    for i in range(objects.ObjectsLength()):
        o = objects.Objects(i)
        b, e, _ = sections[i]
        new_end = b + shifts[i] + (len(new_sections[i]) if i in new_sections else e - b)
        for slot, value in ((6, b + shifts[i]), (8, new_end)):
            struct.pack_into("<Q", header, o._tab.Pos + o._tab.Offset(slot), value)
    f.seek(0)
    f.write(header)
mode = "split" if SPLIT else (f"{shift:+g} int" if INT_MODE else f"{shift:+g} float")
print(f"wrote {dst} ({len(new_sections)} sections patched, mode {mode})")
