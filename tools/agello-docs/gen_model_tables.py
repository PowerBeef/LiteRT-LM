#!/usr/bin/env python3
"""generated/models/<key>.{md,json}: what is inside each .litertlm file, from its bytes.

Run in the tools venv (~/.venvs/agello-tools): needs ai_edge_litert, litert_lm_builder, flatbuffers.
  gen_model_tables.py                  local files and remote files in pins.json
  gen_model_tables.py --only <key>...  some of them

Per file: container version, every section (offsets, size, data type, key/value items), the
metadata protos as text, and for every TFLite section and signature: inputs and outputs, op
histogram, composites and their attributes, weight bytes by tensor type, and, for the prefill/decode
section, the per-layer attention table from model-surgery/inspect_e4b.py.

Remote files (pins.json "remote_models"):
  - "graphs": false: read with HTTP range requests (header and metadata sections only).
  - "graphs": true: downloaded into $SCRATCH (default $TMPDIR), hashed, analyzed, then deleted.
    Some files keep weights inline in the flatbuffer (E2B's per-layer embedder and prefill/decode),
    so their graphs can't be read without the weights.
"""
import collections
import contextlib
import io
import json
import os
import hashlib
import re
import shutil
import struct
import sys
import tempfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # also under python -I
import common  # noqa: E402

sys.path.insert(0, os.path.join(common.HERE, "..", "model-surgery"))
import inspect_e4b as ie  # noqa: E402
from ai_edge_litert import schema_py_generated as S  # noqa: E402
from google.protobuf import text_format  # noqa: E402
from litert_lm_builder import litertlm_header_schema_py_generated as H  # noqa: E402
from litert_lm_builder import litertlm_peek  # noqa: E402
from litert_lm_builder.runtime.proto import embedding_metadata_pb2, executor_metadata_pb2, llm_metadata_pb2  # noqa: E402

DATA_TYPES = {v: k for k, v in vars(H.AnySectionDataType).items() if not k.startswith("_")}
PROTOS = {"LlmMetadataProto": llm_metadata_pb2.LlmMetadata, "ExecutorMetadataProto": executor_metadata_pb2.ExecutorMetadata,
          "EmbeddingMetadataProto": embedding_metadata_pb2.EmbeddingMetadata}
GRAPH_CAP = 256 << 20


class LocalReader:
    def __init__(self, path):
        self.path, self.size = path, os.path.getsize(path)

    def read(self, offset, length):
        with open(self.path, "rb") as f:
            f.seek(offset)
            return f.read(length)


class RemoteReader:
    def __init__(self, url):
        self.url, self.fetched = url, 0
        self.size = int(self._get(0, 1, head=True))

    def _get(self, offset, length, head=False):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={offset}-{offset + length - 1}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            if head:
                return r.headers["Content-Range"].split("/")[1]
            data = r.read()
        self.fetched += len(data)
        return data

    def read(self, offset, length):
        return self._get(offset, length)


def header(reader):
    head = reader.read(0, 32)
    major, minor, patch = struct.unpack("<III", head[8:20])
    end = struct.unpack("<Q", head[24:32])[0]
    md = H.LiteRTLMMetaData.GetRootAs(reader.read(0, end), 32)
    sections = []
    for i in range(md.SectionMetadata().ObjectsLength()):
        o = md.SectionMetadata().Objects(i)
        items = {}
        for j in range(o.ItemsLength()):
            d = litertlm_peek.kvp_to_dict(o.Items(j))
            items[d.get("key")] = d.get("value")
        sections.append({"index": i, "begin": o.BeginOffset(), "end": o.EndOffset(),
                         "size": o.EndOffset() - o.BeginOffset(),
                         "data_type": DATA_TYPES.get(o.DataType(), o.DataType()), "items": items})
    system = {}
    sm = md.SystemMetadata()
    for i in range(sm.EntriesLength() if sm else 0):
        d = litertlm_peek.kvp_to_dict(sm.Entries(i))
        system[d.get("key")] = d.get("value")
    return {"version": f"{major}.{minor}.{patch}", "header_bytes": end, "system": system, "sections": sections}


def load_graph(reader, section):
    """The section's flatbuffer up to its first external buffer (weights after the graph stay unread)."""
    cap = GRAPH_CAP if isinstance(reader, RemoteReader) else section["size"]
    want = min(section["size"], 8 << 20)
    while True:
        data = reader.read(section["begin"], want)
        try:
            model = S.Model.GetRootAs(data, 0)
            external = {i: (model.Buffers(i).Offset(), model.Buffers(i).Size())
                        for i in range(model.BuffersLength()) if model.Buffers(i).Offset() > 1}
            graph_bytes = min(o for o, _ in external.values()) if external else section["size"]
            if graph_bytes <= len(data):
                mt = S.ModelT.InitFromObj(S.Model.GetRootAs(data[:graph_bytes], 0))
                return mt, graph_bytes, external
            want = graph_bytes
        except (IndexError, struct.error, ValueError):
            if want >= cap:
                raise
            want = min(cap, want * 2)


def tensor_type(gr, index):
    x = gr.g.tensors[index]
    dims = x.shapeSignature if x.shapeSignature is not None and len(x.shapeSignature) else x.shape
    return f"{ie.TYPES.get(x.type, x.type)}{[int(d) for d in (dims if dims is not None else [])]}"


def tflite_summary(reader, section):
    mt, graph_bytes, external = load_graph(reader, section)
    # Weights live inline, after the graph in the same section (Buffer.offset > 1), or in a separate
    # TFLiteWeights section referenced by Tensor.external_buffer IDs (Model.external_buffers).
    # Each signature's subgraph has its own IDs; IDs of different subgraphs can share bytes.
    ext_ids = {e.id: (e.group, e.offset, e.length) for e in (mt.externalBuffers or [])}
    by_type = collections.defaultdict(lambda: {"inline": 0, "external": 0, "weights_section": 0,
                                               "tensors": 0, "per_channel": 0})
    seen = set()
    for g in mt.subgraphs:
        for t in g.tensors:
            ext = getattr(t, "externalBuffer", 0) or 0
            if ext:
                group, offset, size = ext_ids.get(ext, (None, None, 0))
                if ("x", group, offset) in seen:
                    continue
                seen.add(("x", group, offset))
                where = "weights_section"
            else:
                if t.buffer in seen or t.buffer == 0 or t.buffer >= len(mt.buffers):
                    continue
                b = mt.buffers[t.buffer]
                size = external[t.buffer][1] if t.buffer in external else (len(b.data) if b.data is not None else 0)
                where = "external" if t.buffer in external else "inline"
                seen.add(t.buffer)
            if not size:
                continue
            row = by_type[ie.TYPES.get(t.type, str(t.type))]
            row[where] += size
            row["tensors"] += 1
            q = t.quantization
            if q is not None and q.scale is not None and len(q.scale) > 1:
                row["per_channel"] += 1

    sigs = []
    for s in mt.signatureDefs:
        gr = ie.Graph(mt, mt.subgraphs[s.subgraphIndex])
        hist = collections.Counter(gr.op(o) for o in gr.g.operators)
        comps = collections.defaultdict(collections.Counter)
        for o in gr.g.operators:
            if gr.op(o).startswith("COMPOSITE:"):
                kind, attrs = gr.composite(o)
                comps[kind][json.dumps(attrs, sort_keys=True, default=str)] += 1
        sigs.append({
            "key": s.signatureKey.decode(), "subgraph": s.subgraphIndex,
            "ops": len(gr.g.operators), "tensors": len(gr.g.tensors),
            "inputs": [[i.name.decode(), tensor_type(gr, i.tensorIndex)] for i in s.inputs],
            "outputs": [[o.name.decode(), tensor_type(gr, o.tensorIndex)] for o in s.outputs],
            "op_histogram": dict(hist.most_common()),
            "composites": {k: {"count": sum(v.values()), "attribute_sets": dict(v.most_common(8)),
                               "distinct": len(v)} for k, v in comps.items()},
        })
    layers = ""
    if section["items"].get("model_type") == "tf_lite_prefill_decode" and any(s["key"] == "decode" for s in sigs):
        sig = next(s for s in mt.signatureDefs if s.signatureKey.decode() == "decode")
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                ie.layer_table(ie.Graph(mt, mt.subgraphs[sig.subgraphIndex]), sig)
            layers = re.sub(r"np\.int\d+\((-?\d+)\)", r"\1", buf.getvalue())
        except Exception as e:  # the table assumes Gemma 4's attention pattern
            layers = f"(layer table not available: {type(e).__name__}: {e})"
    return {"graph_bytes": graph_bytes, "external_bytes": sum(s for _, s in external.values()),
            "weights_section_bytes": sum(length for (_, _, length) in set(ext_ids.values())),
            "weights_by_type": dict(by_type), "signatures": sigs, "layer_table": layers}


def metadata_text(reader, info):
    """litertlm_peek's output for local files; the metadata protos decoded as text for remote ones."""
    if isinstance(reader, LocalReader):
        buf = io.StringIO()
        litertlm_peek.peek_litertlm_file(reader.path, None, buf)
        return buf.getvalue()
    out = []
    for s in info["header"]["sections"]:
        proto = PROTOS.get(s["data_type"])
        if proto and s["size"] < (4 << 20):
            m = proto()
            m.ParseFromString(reader.read(s["begin"], s["size"]))
            out.append(f"Section {s['index']} {s['data_type']}:\n{text_format.MessageToString(m)}")
    return "\n".join(out)


def render(key, info):
    md = [common.gen_header("gen_model_tables.py", [info.get("source", key)]), f"# Model file: {key}\n\n",
          f"- File: `{info['file']}`, {info['bytes']:,} bytes"
          + (f", sha256 `{info['sha256']}`" if info.get("sha256") else "") + "\n",
          f"- Source: {info.get('source', 'local')}\n",
          f"- Container version {info['header']['version']}, header {info['header']['header_bytes']:,} bytes\n"]
    if info.get("remote_bytes_read"):
        md.append(f"- Read remotely: {info['remote_bytes_read'] / 1e6:,.1f} MB fetched (header, metadata and graphs)\n")
    if info["header"]["system"]:
        md.append("- System metadata: " + ", ".join(f"`{k}` = `{v}`" for k, v in info["header"]["system"].items()) + "\n")
    md.append("\n## Sections\n\n| # | model_type / role | Data type | Size (MB) | Items |\n| --- | --- | --- | --- | --- |\n")
    for s in info["header"]["sections"]:
        items = {k: v for k, v in s["items"].items() if k != "model_type"}
        md.append(f"| {s['index']} | {s['items'].get('model_type', '')} | {s['data_type']} | {s['size'] / 1e6:,.1f} "
                  f"| {', '.join(f'{k}={v}' for k, v in items.items())} |\n")
    for s in info["header"]["sections"]:
        t = info["tflite"].get(str(s["index"]))
        if not t:
            continue
        if "error" in t:
            md.append(f"\n## Section {s['index']}: {s['items'].get('model_type', s['data_type'])}\n\n(not parsed: {t['error']})\n")
            continue
        md.append(f"\n## Section {s['index']}: {s['items'].get('model_type', s['data_type'])}\n\n"
                  f"Graph {t['graph_bytes'] / 1e6:,.1f} MB; weights after the graph {t['external_bytes'] / 1e6:,.1f} MB; "
                  f"weights in a TFLiteWeights section {t['weights_section_bytes'] / 1e6:,.1f} MB.\n\n"
                  "| Tensor type | Tensors | Inline MB | After graph MB | Weights section MB | Per-channel quantized |\n"
                  "| --- | --- | --- | --- | --- | --- |\n")
        for typ, r in sorted(t["weights_by_type"].items(),
                             key=lambda kv: -(kv[1]["inline"] + kv[1]["external"] + kv[1]["weights_section"])):
            md.append(f"| {typ} | {r['tensors']} | {r['inline'] / 1e6:,.1f} | {r['external'] / 1e6:,.1f} "
                      f"| {r['weights_section'] / 1e6:,.1f} | {r['per_channel']} |\n")
        for sig in t["signatures"]:
            md.append(f"\n### Signature `{sig['key']}` (subgraph {sig['subgraph']}): {sig['ops']} ops, {sig['tensors']} tensors\n\n")
            md.append("| Input | Type and shape |\n| --- | --- |\n" + "".join(f"| `{n}` | `{x}` |\n" for n, x in sig["inputs"]))
            md.append("\n| Output | Type and shape |\n| --- | --- |\n" + "".join(f"| `{n}` | `{x}` |\n" for n, x in sig["outputs"]))
            md.append("\nOps: " + ", ".join(f"{k} {v}" for k, v in sig["op_histogram"].items()) + "\n")
            if sig["composites"]:
                md.append("\n| Composite | Count | Distinct attribute sets | Most common attributes |\n| --- | --- | --- | --- |\n")
                for k, c in sig["composites"].items():
                    top = next(iter(c["attribute_sets"]))
                    md.append(f"| `{k}` | {c['count']} | {c['distinct']} | `{top}` |\n")
        if t["layer_table"]:
            md.append("\n### Attention per layer (`decode`)\n\n```\n" + t["layer_table"].strip() + "\n```\n")
    md.append("\n## Metadata\n\n<details><summary>Metadata sections as text</summary>\n\n```\n"
              + info["metadata"].strip() + "\n```\n\n</details>\n")
    return "".join(md)


def analyze(key, reader, meta, graphs=True):
    info = {"file": os.path.basename(meta.get("path") or meta.get("url")), "bytes": reader.size,
            **{k: v for k, v in meta.items() if k in ("sha256", "source")}, "header": header(reader), "tflite": {}}
    info["metadata"] = metadata_text(reader, info)
    for s in info["header"]["sections"]:
        if graphs and s["data_type"] == "TFLiteModel":
            print(f"  {key}: section {s['index']} {s['items'].get('model_type')}", file=sys.stderr)
            try:
                info["tflite"][str(s["index"])] = tflite_summary(reader, s)
            except Exception as e:
                info["tflite"][str(s["index"])] = {"error": f"{type(e).__name__}: {e}"}
    if isinstance(reader, RemoteReader):
        info["remote_bytes_read"] = reader.fetched
    return info


def download(url, expected_bytes):
    """Stream a remote file into a scratch folder, hashing it; the caller deletes the folder."""
    root = os.environ.get("SCRATCH") or tempfile.gettempdir()
    free = shutil.disk_usage(root).free
    if free < expected_bytes + (10 << 30):
        sys.exit(f"not enough disk for {expected_bytes / 1e9:.1f} GB in {root} ({free / 1e9:.0f} GB free)")
    folder = tempfile.mkdtemp(prefix="agello-docs-model-", dir=root)
    path = os.path.join(folder, os.path.basename(url))
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=600) as r, open(path, "wb") as f:
        while chunk := r.read(8 << 20):
            f.write(chunk)
            digest.update(chunk)
    return folder, path, digest.hexdigest()


def main():
    targets = {k: (LocalReader, m["path"], m, True) for k, m in common.PINS["models"].items()}
    targets.update({k: (RemoteReader, m["url"], m, m.get("graphs", True))
                    for k, m in common.PINS.get("remote_models", {}).items()})
    if len(sys.argv) > 2 and sys.argv[1] == "--only":
        targets = {k: targets[k] for k in sys.argv[2:]}
    out = os.path.join(common.GEN_DIR, "models")
    os.makedirs(out, exist_ok=True)
    for key, (cls, where, meta, graphs) in targets.items():
        folder = None
        if cls is RemoteReader and graphs:
            size = RemoteReader(where).size
            print(f"  {key}: downloading {size / 1e9:.2f} GB", file=sys.stderr)
            folder, path, sha = download(where, size)
            meta = {**meta, "sha256": sha, "path": path}
            cls, where = LocalReader, path
        try:
            info = analyze(key, cls(where), meta, graphs)
            if folder:
                info["file"] = os.path.basename(meta["url"])
        finally:
            if folder:
                shutil.rmtree(folder, ignore_errors=True)
        json.dump({k: v for k, v in info.items() if k != "metadata"}, open(os.path.join(out, f"{key}.json"), "w"),
                  indent=1, default=str)
        with open(os.path.join(out, f"{key}.md"), "w") as f:
            f.write(render(key, info))
        print(os.path.join(out, f"{key}.md") + (f"  ({info['remote_bytes_read'] / 1e6:.1f} MB fetched)"
                                                 if info.get("remote_bytes_read") else ""))


if __name__ == "__main__":
    main()
