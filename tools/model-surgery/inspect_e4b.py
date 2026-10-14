"""Map the attention structure of a Gemma 4 .litertlm (spike S4, docs/agello/E4B-GRAPH.md).

For a TFLite section (default: tf_lite_prefill_decode) and one of its
signatures (default: decode), prints:
  - the signature's inputs and outputs;
  - an op histogram and every composite kind with its attributes;
  - for each SOFTMAX: its layer, input shape, how it is masked (the ops and
    graph inputs behind the mask operand) and which KV-cache inputs feed it.

Usage (in the tools venv, ~/.venvs/agello-tools):
  python inspect_e4b.py <model.litertlm> [--section tf_lite_mtp_drafter] [--signature decode]
"""
import argparse
import collections
import re
import struct

from ai_edge_litert import schema_py_generated as S
from flatbuffers import flexbuffers
from litert_lm_builder import litertlm_header_schema_py_generated as H

OPS = {v: k for k, v in vars(S.BuiltinOperator).items() if not k.startswith("_")}
TYPES = {v: k for k, v in vars(S.TensorType).items() if not k.startswith("_")}


def sections(path):
    with open(path, "rb") as f:
        header_end = struct.unpack("<Q", f.read(32)[24:32])[0]
        f.seek(0)
        header = f.read(header_end)
    md = H.LiteRTLMMetaData.GetRootAs(header, 32)
    out = []
    for i in range(md.SectionMetadata().ObjectsLength()):
        o = md.SectionMetadata().Objects(i)
        kind = None
        for j in range(o.ItemsLength()):
            kv = o.Items(j)
            if kv.Key().decode() == "model_type":
                v = H.StringValue()
                v.Init(kv.Value().Bytes, kv.Value().Pos)
                kind = v.Value().decode()
        out.append((o.BeginOffset(), o.EndOffset(), kind))
    return out


def load_section(path, wanted):
    begin, end, _ = next(s for s in sections(path) if s[2] == wanted)
    with open(path, "rb") as f:
        f.seek(begin)
        data = f.read(end - begin)
    model = S.Model.GetRootAs(data, 0)
    external = [model.Buffers(i).Offset() for i in range(model.BuffersLength()) if model.Buffers(i).Offset() > 1]
    graph = data[: min(external)] if external else data
    return S.ModelT.InitFromObj(S.Model.GetRootAs(graph, 0))


def name(t):
    return t.name.decode() if isinstance(t.name, bytes) else t.name


class Graph:
    def __init__(self, mt, g):
        self.mt, self.g = mt, g
        self.producer, self.consumers = {}, collections.defaultdict(list)
        for o in g.operators:
            for t in o.outputs:
                self.producer[int(t)] = o
            for t in o.inputs:
                if t >= 0:
                    self.consumers[int(t)].append(o)
        self.inputs = {int(t) for t in g.inputs}

    def op(self, o):
        c = self.mt.operatorCodes[o.opcodeIndex]
        code = max(c.builtinCode, c.deprecatedBuiltinCode)
        if code == S.BuiltinOperator.CUSTOM:
            return "CUSTOM:" + c.customCode.decode()
        if code == S.BuiltinOperator.STABLEHLO_COMPOSITE:
            return "COMPOSITE:" + self.composite(o)[0]
        return OPS.get(code, str(code))

    def composite(self, o):
        opts = o.builtinOptions2
        attrs = {}
        if opts is not None and opts.compositeAttributes is not None and len(opts.compositeAttributes):
            attrs = flexbuffers.Loads(bytes(opts.compositeAttributes))
        return opts.name.decode() if opts is not None else "?", attrs

    def tensor(self, t):
        x = self.g.tensors[t]
        shape = list(x.shapeSignature) if x.shapeSignature is not None and len(x.shapeSignature) else list(x.shape)
        return f"{name(x)} {TYPES.get(x.type, x.type)}{shape}"

    def closure(self, start, limit=400):
        """Ops and graph inputs behind tensor `start` (bounded walk)."""
        seen, ops, inputs, stack = set(), [], set(), [start]
        while stack and len(ops) < limit:
            t = stack.pop()
            if t in seen or t < 0:
                continue
            seen.add(t)
            if t in self.inputs:
                inputs.add(name(self.g.tensors[t]))
            o = self.producer.get(t)
            if o is None:
                continue
            ops.append(o)
            stack.extend(int(i) for i in o.inputs)
        return ops, inputs


def layer_of(text):
    m = re.search(r"(?:layer|block|layers)[_./]?(\d+)", text)
    return int(m.group(1)) if m else None


def layer_table(gr, sig):
    """One row per attention: the caches it reads, their scales, and its mask."""
    names = {o.tensorIndex: o.name.decode() for o in sig.outputs}
    names.update({i.tensorIndex: i.name.decode() for i in sig.inputs})

    def cache(t):
        # A cache is a signature input, or the updated copy the graph outputs.
        if t in names:
            return names[t]
        o = gr.producer.get(t)
        while o is not None and not gr.op(o).startswith("COMPOSITE:odml.cache_update"):
            o = gr.producer.get(int(o.inputs[0]))
        return names.get(int(o.outputs[0]), "?") if o is not None else "?"

    print("\n== attention per layer: layer, kind, K, V, K scale, V scale, mask, softmax shape")
    for o in gr.g.operators:
        if gr.op(o) != "SOFTMAX":
            continue
        select = gr.producer[int(o.inputs[0])]
        qk = gr.producer[int(select.inputs[1])]
        pv = next(q for q in gr.consumers[int(o.outputs[0])] if gr.op(q).startswith("COMPOSITE"))
        _, qk_attrs = gr.composite(qk)
        _, pv_attrs = gr.composite(pv)
        layer = layer_of(name(gr.g.tensors[int(qk.inputs[0])]))
        ops, inputs = gr.closure(int(select.inputs[0]))
        kinds = {gr.op(q) for q in ops}
        mask = "+".join(sorted(i.split(":")[0].split("_", 1)[-1] for i in inputs))
        if "LESS" in kinds:
            mask = f"window({mask})"
        shape = list(gr.g.tensors[int(o.inputs[0])].shape)
        print(f"  {layer:>3} {'global' if qk_attrs['is_global'] else 'local ':<6} {cache(int(qk.inputs[1])):<14}"
              f" {cache(int(pv.inputs[1])):<14} {qk_attrs['scale']:.6f} {pv_attrs['scale']:.6f} {mask:<24} {shape}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--section", default="tf_lite_prefill_decode")
    p.add_argument("--signature", default="decode")
    p.add_argument("--names", action="store_true", help="print tensor names around each softmax")
    p.add_argument("--layers", action="store_true", help="one row per attention instead of the softmax walk")
    a = p.parse_args()

    mt = load_section(a.model, a.section)
    sigs = {s.signatureKey.decode(): s for s in mt.signatureDefs}
    print(f"signatures: {', '.join(f'{k} (subgraph {s.subgraphIndex})' for k, s in sigs.items())}")
    sig = sigs[a.signature]
    gr = Graph(mt, mt.subgraphs[sig.subgraphIndex])
    print(f"\n== {a.section} / {a.signature}: {len(gr.g.operators)} ops, {len(gr.g.tensors)} tensors")
    print("inputs:")
    for i in sig.inputs:
        print(f"  {i.name.decode():<28} {gr.tensor(i.tensorIndex)}")
    print("outputs:")
    for o in sig.outputs:
        print(f"  {o.name.decode():<28} {gr.tensor(o.tensorIndex)}")

    hist = collections.Counter(gr.op(o) for o in gr.g.operators)
    print("\nops:", ", ".join(f"{k} {v}" for k, v in hist.most_common()))

    kinds = {}
    for o in gr.g.operators:
        if gr.op(o).startswith("COMPOSITE:"):
            kind, attrs = gr.composite(o)
            kinds.setdefault(kind, collections.Counter())[repr(sorted(attrs.items()))] += 1
    print("\ncomposites (attributes: count):")
    for kind, counter in kinds.items():
        for attrs, n in counter.most_common():
            print(f"  {kind}: {attrs}: {n}")

    if a.layers:
        layer_table(gr, sig)
        return
    print("\n== softmaxes")
    for o in gr.g.operators:
        if gr.op(o) != "SOFTMAX":
            continue
        x = int(o.inputs[0])
        pre = gr.producer.get(x)
        layer = layer_of(name(gr.g.tensors[x]))
        line = f"layer {layer}: softmax in {gr.tensor(x)}; made by {gr.op(pre) if pre else 'input'}"
        if pre is not None:
            for operand in pre.inputs:
                ops, inputs = gr.closure(int(operand))
                kinds_behind = collections.Counter(gr.op(q) for q in ops)
                caches = sorted(i for i in inputs if "cache" in i)
                line += (f"\n    operand {gr.tensor(int(operand))}: {len(ops)} ops behind"
                         f" ({', '.join(f'{k} {v}' for k, v in kinds_behind.most_common(8))});"
                         f" inputs {sorted(i for i in inputs if 'cache' not in i)}"
                         + (f"; caches {caches}" if caches else ""))
        print(line)
        if a.names:
            for q in gr.consumers.get(int(o.outputs[0]), []):
                print("    consumer:", gr.op(q), [gr.tensor(int(t)) for t in q.inputs])


if __name__ == "__main__":
    main()
