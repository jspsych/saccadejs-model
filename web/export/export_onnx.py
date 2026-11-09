import argparse
import collections
import datetime
import json
import os
import re
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from download_checkpoint import ENTITY, PROJECT, RUN, download

HERE = Path(__file__).resolve().parent
# Build outputs live under web/export/out/ (gitignored). web/export/parity/ reaches
# them through symlinks; the saccade.js repo takes its copies from here explicitly.
OUT_DIR = HERE / "out"
MODELS_DIR = OUT_DIR / "models"
GEN_DIR = OUT_DIR / "generated"

IN_NAME = "eye_image"
OUT_NAME = "embedding"
# Second graph output: the trained per-image calibration weight. saccade.js reads it by this
# name and uses it to weight both the calibration rows and the smoothing mean. Exporting it
# with the model is what keeps it bound to the embedding space it was trained against --
# shipping it as separate JSON let the two drift apart, since the model is fetched by URL at
# runtime while a JSON file is frozen into whatever @saccadejs/core version is installed.
WEIGHT_NAME = "cal_weight"
IN_SHAPE = (1, 36, 144, 1)
OPSET = 17
# Initializers at or below this many elements are named positionally by
# canonicalize(); see the comment there.
SMALL_CONST = 16


def build_models(weights_path):
    from et_util.config import TrainConfig
    from et_util.models import create_masked_model

    model = create_masked_model(TrainConfig())
    model.load_weights(str(weights_path))
    return model, model.get_layer("Image_Embeddings").layer, model.get_layer(
        "Calibration_Weights")


def build_export_model(emb, cal):
    """One crop in, (embedding, cal_weight) out.

    The calibration head is already part of the trained model -- ``Dense(1, sigmoid)`` applied
    through SimpleTimeDistributed, so it scores one image's embedding, exactly what a browser
    has per frame. This wrapper just re-uses the trained layer on a single image instead of a
    (batch, targets, dim) stack, so the exported graph carries the weights it was trained with.

    A set-level head (``cfg.calibration_weighting == "set_attention"``) cannot come along:
    CalibrationWeightingHead attends across the whole calibration set and optionally reads each
    point's coordinates, and the browser runs this graph on one crop at a time with neither.
    Weighting like that has to run once over the collected set at fit time, which is a
    different artefact than this one -- so refuse rather than export something that looks
    right and silently scores each frame out of context.
    """
    import keras

    if not isinstance(cal, keras.layers.Dense):
        raise SystemExit(
            f"Calibration_Weights is {type(cal).__name__}, not a pointwise Dense head. A "
            "set-level head cannot be a second output of a per-crop graph: it needs every "
            "calibration point at once, which the browser does not have per frame. Export "
            "with calibration_weighting=\"linear\", or build the set-level weighter as its "
            "own model invoked at fit time.")

    from et_util.config import IMAGE_SHAPE

    inp = keras.layers.Input(shape=IMAGE_SHAPE, name=IN_NAME)
    embedding = emb(inp)
    weight = cal(embedding)
    return keras.Model(inp, [embedding, weight], name="EyeEmbeddingWithCalWeight")


def fix_erfc(m):
    """TF 2.20 exact-gelu emits Erfc, which tf2onnx 1.16 leaves unconverted.

    Erfc(x) == 1 - Erf(x).
    """
    from onnx import helper, numpy_helper

    g = m.graph
    targets = [i for i, n in enumerate(g.node) if n.op_type == "Erfc"]
    if not targets:
        return 0
    g.initializer.append(numpy_helper.from_array(
        np.ones((), dtype=np.float32), "erfc_one"))
    for i in reversed(targets):
        n = g.node[i]
        erf_out = n.output[0] + "_erf"
        new = [
            helper.make_node("Erf", [n.input[0]], [erf_out], name=n.name + "_erf"),
            helper.make_node("Sub", ["erfc_one", erf_out], [n.output[0]],
                             name=n.name + "_sub"),
        ]
        del g.node[i]
        for k, node in enumerate(new):
            g.node.insert(i + k, node)
    return len(targets)


def _graph_index(g):
    from onnx import numpy_helper

    prod = {o: n for n in g.node for o in n.output}
    uses = collections.Counter(i for n in g.node for i in n.input)
    uses.update(o.name for o in g.output)
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    return prod, uses, consts


def _pnode(prod, tensor, op_type):
    n = prod.get(tensor)
    return n if n is not None and n.op_type == op_type else None


def _scalar(consts, tensor, value=None, tol=1e-6):
    v = consts.get(tensor)
    if v is None or v.size != 1:
        return None
    f = float(v.reshape(-1)[0])
    if value is not None and abs(f - value) > tol:
        return None
    return f


def _splice(g, repls):
    """repls: list of (anchor_node, chain_nodes, new_nodes). Chains are disjoint."""
    drop = {id(n) for _, chain, _ in repls for n in chain}
    insert = {id(a): new for a, _, new in repls}
    out = []
    for n in g.node:
        if id(n) in insert:
            out.extend(insert[id(n)])
        elif id(n) in drop:
            continue
        else:
            out.append(n)
    del g.node[:]
    g.node.extend(out)


def _chain_is_private(chain, anchor, uses):
    """Every chain tensor except the anchor's output is consumed only inside the chain."""
    internal = collections.Counter(i for n in chain for i in n.input)
    return all(uses[t] == internal[t] for n in chain if n is not anchor for t in n.output)


def fuse_gelu(m):
    """0.5*x*Erfc(-x/sqrt(2)) == Gelu(x) -- com.microsoft.Gelu, which ORT Web's
    WebGPU EP implements (it is also what ORT's own GeluFusion emits)."""
    from onnx import helper

    g = m.graph
    prod, uses, consts = _graph_index(g)
    repls, claimed = [], set()
    for n in g.node:
        if n.op_type != "Mul":
            continue
        for a, b in ((0, 1), (1, 0)):
            half = _pnode(prod, n.input[a], "Mul")
            erfc = _pnode(prod, n.input[b], "Erfc")
            if half is None or erfc is None:
                continue
            x = next((half.input[j] for i, j in ((0, 1), (1, 0))
                      if _scalar(consts, half.input[i], 0.5) is not None), None)
            mulc = _pnode(prod, erfc.input[0], "Mul")
            if x is None or mulc is None:
                continue
            neg = next((_pnode(prod, mulc.input[j], "Neg") for i, j in ((0, 1), (1, 0))
                        if _scalar(consts, mulc.input[i], 2 ** -0.5) is not None), None)
            if neg is None or neg.input[0] != x:
                continue
            chain = [neg, mulc, erfc, half, n]
            if not _chain_is_private(chain, n, uses) or claimed & {id(c) for c in chain}:
                continue
            claimed |= {id(c) for c in chain}
            repls.append((n, chain, [helper.make_node(
                "Gelu", [x], [n.output[0]], name=n.name + "_gelu",
                domain="com.microsoft")]))
            break
    _splice(g, repls)
    if repls:
        m.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    return len(repls)


def fuse_layernorm(m):
    """Rewrite tf2onnx's 13/14-node moments chain as one opset-17 LayerNormalization.

    TF emits y = x*s + (beta - mean*s), s = gamma/sqrt(var+eps). The conv-side norms
    reduce the channel axis of an NCHW tensor, which ONNX LayerNormalization cannot
    express (it normalizes the suffix [axis, rank)), so those get NHWC transposes --
    ORT Web's WebGPU EP prefers NHWC for Conv and cancels most of them at load.
    """
    from onnx import helper, numpy_helper

    g = m.graph
    prod, uses, consts = _graph_index(g)
    repls, claimed, new_inits = [], set(), []

    for n in g.node:
        if n.op_type != "Add":
            continue
        for a, b in ((0, 1), (1, 0)):
            mul2 = _pnode(prod, n.input[a], "Mul")
            add1 = _pnode(prod, n.input[b], "Add")
            if mul2 is None or add1 is None:
                continue
            hit = next(((consts[add1.input[i]], _pnode(prod, add1.input[j], "Mul"))
                        for i, j in ((0, 1), (1, 0))
                        if add1.input[i] in consts
                        and _pnode(prod, add1.input[j], "Mul") is not None), None)
            if hit is None:
                continue
            beta, mul1 = hit
            shared = set(mul1.input) & set(mul2.input)
            if len(shared) != 1:
                continue
            scale = shared.pop()
            neg = _pnode(prod, next(t for t in mul1.input if t != scale), "Neg")
            x = next(t for t in mul2.input if t != scale)
            snode = _pnode(prod, scale, "Mul")
            if neg is None or snode is None:
                continue
            hit = next(((consts[snode.input[i]], snode.input[j])
                        for i, j in ((0, 1), (1, 0)) if snode.input[i] in consts), None)
            if hit is None:
                continue
            gamma, invt = hit
            inv = _pnode(prod, invt, "Reciprocal")
            sqrt = _pnode(prod, inv.input[0], "Sqrt") if inv else None
            adde = _pnode(prod, sqrt.input[0], "Add") if sqrt else None
            if adde is None:
                continue
            hit = next(((_scalar(consts, adde.input[i]), adde.input[j])
                        for i, j in ((0, 1), (1, 0))
                        if _scalar(consts, adde.input[i]) is not None), None)
            if hit is None:
                continue
            eps, vart = hit
            var = prod.get(vart)
            if var is None or var.op_type not in ("ReduceMean", "GlobalAveragePool"):
                continue
            sqd = _pnode(prod, var.input[0], "Mul")
            if sqd is None or sqd.input[0] != sqd.input[1]:
                continue
            d = _pnode(prod, sqd.input[0], "Sub")
            if d is None or d.input[0] != x:
                continue
            rsh = _pnode(prod, d.input[1], "Reshape")
            meant = rsh.input[0] if rsh is not None else d.input[1]
            mean = prod.get(meant)
            if (mean is None or mean.op_type != var.op_type
                    or mean.input[0] != x or neg.input[0] != meant):
                continue
            # gamma/beta keep their broadcast shape (e.g. [1,C,1,1]), which pins both
            # the rank of x and the normalized axis without needing shape inference
            # (it gives up partway through the einsum-rewritten attention blocks).
            rank = gamma.ndim
            if beta.shape != gamma.shape or rank < 2:
                continue
            nonunit = [i for i, s in enumerate(gamma.shape) if s != 1]
            if len(nonunit) != 1 or gamma.size < 2:
                continue
            axis = nonunit[0]
            if mean.op_type == "GlobalAveragePool":
                if rank != 3 or axis != 2:
                    continue
            else:
                attrs = {at.name: at for at in mean.attribute}
                if attrs.get("keepdims") is None or attrs["keepdims"].i != 1:
                    continue
                axes = list(attrs["axes"].ints) if "axes" in attrs else []
                if len(axes) != 1 or axes[0] % rank != axis:
                    continue
            if rsh is not None:
                tgt = consts.get(rsh.input[1])
                if tgt is None or len(tgt) != rank or tgt[axis] != 1:
                    continue

            chain = [mean, d, sqd, var, adde, sqrt, inv, snode, neg, mul1, add1, mul2, n]
            if rsh is not None:
                chain.insert(1, rsh)
            if claimed & {id(c) for c in chain} or not _chain_is_private(chain, n, uses):
                continue

            tag = n.name.replace("/", "_")
            gn, bn = tag + "_ln_gamma", tag + "_ln_beta"
            new_inits += [
                numpy_helper.from_array(gamma.reshape(-1).astype(np.float32), gn),
                numpy_helper.from_array(beta.reshape(-1).astype(np.float32), bn)]
            if axis == rank - 1:
                new = [helper.make_node("LayerNormalization", [x, gn, bn], [n.output[0]],
                                        name=tag + "_ln", axis=-1, epsilon=eps)]
            elif rank == 4 and axis == 1:
                t1, t2 = tag + "_nhwc", tag + "_ln_out"
                new = [
                    helper.make_node("Transpose", [x], [t1], name=tag + "_to_nhwc",
                                     perm=[0, 2, 3, 1]),
                    helper.make_node("LayerNormalization", [t1, gn, bn], [t2],
                                     name=tag + "_ln", axis=-1, epsilon=eps),
                    helper.make_node("Transpose", [t2], [n.output[0]],
                                     name=tag + "_to_nchw", perm=[0, 3, 1, 2]),
                ]
            else:
                continue
            claimed |= {id(c) for c in chain}
            repls.append((n, chain, new))
            break

    _splice(g, repls)
    g.initializer.extend(new_inits)
    return len(repls)


def prune_initializers(m):
    live = {i for n in m.graph.node for i in n.input}
    keep = [i for i in m.graph.initializer if i.name in live]
    dropped = len(m.graph.initializer) - len(keep)
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    return dropped


def ort_optimize(src, dst):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(dst)
    ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])


def _trailing_dim(vi):
    """Last dimension of a graph value's declared shape, or None if it is not static."""
    dims = vi.type.tensor_type.shape.dim
    if not dims:
        return None
    d = dims[-1]
    return d.dim_value if d.HasField("dim_value") else None


def rename_io(path, in_name=IN_NAME, out_name=OUT_NAME, weight_name=WEIGHT_NAME):
    """Give the graph its published I/O names.

    Outputs are matched by shape, not position: nothing in Keras export, tf2onnx or the ORT
    optimiser promises to keep two outputs in the order the Keras model declared them, and
    silently swapping the embedding and the weight would produce a model that runs and is
    wrong. The weight is the width-1 output; the embedding is the other one.
    """
    import onnx

    m = onnx.load(str(path)) if not hasattr(path, "graph") else path
    g = m.graph
    old_in = g.input[0].name
    if len(g.output) == 1:
        outs = [(g.output[0], out_name)]
    elif len(g.output) == 2:
        widths = [_trailing_dim(o) for o in g.output]
        scalars = [i for i, w in enumerate(widths) if w == 1]
        if len(scalars) != 1:
            raise SystemExit(
                f"cannot tell the embedding from the calibration weight: output widths "
                f"{widths}. Exactly one output must be width 1.")
        wi = scalars[0]
        outs = [(g.output[wi], weight_name), (g.output[1 - wi], out_name)]
    else:
        raise SystemExit(f"expected 1 or 2 graph outputs, found {len(g.output)}")

    ren = {old_in: in_name}
    for vi, name in outs:
        ren[vi.name] = name
    g.input[0].name = in_name
    for vi, name in outs:
        vi.name = name
    old_out = next(old for old, new in ren.items() if new == out_name)
    for node in g.node:
        for i, v in enumerate(node.input):
            if v in ren:
                node.input[i] = ren[v]
        for i, v in enumerate(node.output):
            if v in ren:
                node.output[i] = ren[v]
    for vi in g.value_info:
        if vi.name in ren:
            vi.name = ren[vi.name]
    onnx.checker.check_model(m)
    return m, old_in, old_out


def _shape_of(vi):
    return [d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in vi.type.tensor_type.shape.dim]


def histogram(m):
    return collections.OrderedDict(
        sorted(collections.Counter(n.op_type for n in m.graph.node).items()))


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def canonicalize(m):
    """Make the serialized bytes a function of the graph, not of the run that produced it.

    Two exports of the same checkpoint used to differ: tf2onnx numbers generated nodes from
    a process-global counter (``Unsqueeze__1071`` here, ``Unsqueeze__684`` there), embeds
    ``id()`` values in einsum-derived names (``einsum5213114608_ba_left_set__9``), labels
    equivalent constants differently between runs, and emits nodes and attributes in an
    order that varies too. None of that changes what the model computes, but all of it
    changes its sha256 -- and the sha256 is what a participant's data records as the model
    they saw. PYTHONHASHSEED does not help; the volatility is not hash randomisation.

    So: sort the graph topologically with a structural signature as the tie-break, replace
    every generated name with its position in that order, and sort each node's attributes.
    Names carrying a TF variable path (they contain "/") are the readable ones, are stable
    across runs already, and are kept.

    This buys byte-identical exports *on one platform*. Across platforms one folded constant
    still differs in the last ULP (58 of 256 elements, ~6e-08 -- TF's own constant folding on
    arm64 versus x86), which is one more reason releases are cut only by CI.
    """
    import hashlib
    import heapq
    import math

    import onnx

    g = m.graph
    nodes = list(g.node)
    keep = {v.name for v in list(g.input) + list(g.output)}
    generated = re.compile(r"__\d+|einsum\d+")

    def h(*parts):
        d = hashlib.sha1()
        for x in parts:
            d.update(x if isinstance(x, bytes) else str(x).encode())
            d.update(b"\x1f")
        return d.hexdigest()

    def content_sig(t):
        c = onnx.TensorProto()
        c.CopyFrom(t)
        c.name = c.doc_string = ""
        return h("t", hashlib.sha1(c.SerializeToString()).hexdigest())

    init_sig = {i.name: content_sig(i) for i in g.initializer}

    # Identical constants (the attention scale appears once per block) get deduplicated
    # somewhere in tf2onnx/ORT, and *which* copy survives varies between runs -- with it,
    # which TF variable path ends up naming the survivor. Merge them ourselves instead,
    # keeping the first name in sort order.
    groups = collections.defaultdict(list)
    for i in g.initializer:
        groups[init_sig[i.name]].append(i.name)
    alias = {}
    for names in groups.values():
        if len(names) > 1:
            keeper, *rest = sorted(names)
            alias.update({r: keeper for r in rest})
    if alias:
        survivors = [i for i in g.initializer if i.name not in alias]
        del g.initializer[:]
        g.initializer.extend(survivors)
        for n in nodes:
            n.input[:] = [alias.get(x, x) for x in n.input]
        init_sig = {k: v for k, v in init_sig.items() if k not in alias}

    graph_in = {i.name: h("in", i.name) for i in g.input}
    producer = {o: (i, k) for i, n in enumerate(nodes) for k, o in enumerate(n.output) if o}
    sig, active = {}, set()

    def tensor_sig(name):
        if not name:
            return "empty"
        if name in init_sig:
            return init_sig[name]
        if name in graph_in:
            return graph_in[name]
        i, k = producer[name]
        return h(node_sig(i), k)

    def node_sig(i):
        if i not in sig:
            if i in active:
                raise SystemExit("cycle in the exported graph")
            active.add(i)
            n = nodes[i]
            attrs = [a.SerializeToString() for a in sorted(n.attribute, key=lambda a: a.name)]
            sig[i] = h("n", n.op_type, *attrs, *[tensor_sig(x) for x in n.input])
        return sig[i]

    for i in range(len(nodes)):
        node_sig(i)

    indeg = [sum(1 for x in n.input if x in producer) for n in nodes]
    consumers = {}
    for i, n in enumerate(nodes):
        for x in n.input:
            if x in producer:
                consumers.setdefault(producer[x][0], []).append(i)
    ready = [(sig[i], i) for i in range(len(nodes)) if indeg[i] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        _, i = heapq.heappop(ready)
        order.append(i)
        for j in consumers.get(i, []):
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(ready, (sig[j], j))
    if len(order) != len(nodes):
        raise SystemExit(f"topological sort covered {len(order)}/{len(nodes)} nodes")

    # Small constants -- attention scales, axes, shape vectors -- exist in many identical
    # copies, and tf2onnx/ORT collapse them upstream of us, keeping an arbitrary one. The
    # survivor carries whichever TF variable path happened to name it, so its "readable"
    # name is not stable either; number them instead. Large tensors are the actual weights,
    # unique by content, and keep the Keras-derived names that make the graph legible.
    small = {i.name for i in g.initializer
             if math.prod(i.dims) <= SMALL_CONST or not i.dims}

    def is_generated(name):
        return bool(name) and name not in keep and ("/" not in name or name in small)

    ren, consts = {}, 0
    for pos, i in enumerate(order):
        n = nodes[i]
        for x in n.input:
            if is_generated(x) and x in init_sig and x not in ren:
                ren[x] = f"const__{consts}"
                consts += 1
        if is_generated(n.name):
            ren[n.name] = f"{n.op_type}__{pos}"
        for k, o in enumerate(n.output):
            if is_generated(o):
                ren[o] = f"{n.op_type}__{pos}:{k}"
    orphans = [i.name for i in g.initializer if is_generated(i.name) and i.name not in ren]
    for name in sorted(orphans, key=lambda x: init_sig[x]):
        ren[name] = f"const__{consts}"
        consts += 1
    if len(set(ren.values())) != len(ren) or set(ren.values()) & keep:
        raise SystemExit("canonical names collided")

    def fix_doc(d):
        # tf2onnx writes prose naming the nodes it rewrote ("Added in transpose optimizer"),
        # so doc_strings carry the volatile names too. Anything left after the mapping names
        # a node that no longer exists.
        if not d or not generated.search(d):
            return d
        for old, new in ren.items():
            if old in d:
                d = d.replace(old, new)
        return generated.sub("__x", d)

    for n in nodes:
        n.doc_string = fix_doc(n.doc_string)
        n.name = ren.get(n.name, n.name)
        n.input[:] = [ren.get(x, x) for x in n.input]
        n.output[:] = [ren.get(x, x) for x in n.output]
        attrs = sorted(n.attribute, key=lambda a: a.name)
        del n.attribute[:]
        n.attribute.extend(attrs)
    for i in g.initializer:
        i.name = ren.get(i.name, i.name)
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        vi.name = ren.get(vi.name, vi.name)

    ordered = [nodes[i] for i in order]
    del g.node[:]
    g.node.extend(ordered)
    inits = sorted(g.initializer, key=lambda i: i.name)
    del g.initializer[:]
    g.initializer.extend(inits)
    return len(ren), len(alias)


def main(fuse=True):
    import tensorflow as tf
    import keras
    import onnx
    import tf2onnx

    weights = download()
    _, emb, cal = build_models(weights)
    exportable = build_export_model(emb, cal)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = HERE / "_direct.onnx"
    exportable.export(
        str(tmp),
        format="onnx",
        input_signature=[tf.TensorSpec(IN_SHAPE, tf.float32, name=IN_NAME)],
        opset_version=OPSET,
    )

    out = MODELS_DIR / "eye_embedding.onnx"
    fixed = onnx.load(str(tmp))
    n_raw = len(fixed.graph.node)
    n_gelu = fuse_gelu(fixed) if fuse else 0
    n_erfc = fix_erfc(fixed)
    n_ln = fuse_layernorm(fixed) if fuse else 0
    if fuse:
        prune_initializers(fixed)
    onnx.checker.check_model(fixed)
    onnx.shape_inference.infer_shapes(fixed)
    tmp2 = HERE / "_fixed.onnx"
    onnx.save(fixed, str(tmp2))
    tmp3 = HERE / "_opt.onnx"
    ort_optimize(tmp2, tmp3)
    m, old_in, old_out = rename_io(tmp3)
    n_canon = canonicalize(m)
    onnx.checker.check_model(m)
    onnx.save(m, str(out))
    for p in (tmp, tmp2, tmp3):
        p.unlink()
    hist = histogram(m)
    domains = sorted({n.domain for n in m.graph.node})
    print(f"nodes: {n_raw} raw -> {len(m.graph.node)} final "
          f"(gelu {n_gelu}, layernorm {n_ln}, erfc {n_erfc}); domains: {domains}")
    outs = ", ".join(f"{o.name}{list(_shape_of(o))}" for o in m.graph.output)
    print(f"io: {old_in} -> {IN_NAME}, {old_out} -> {OUT_NAME}; outputs: {outs}")
    print(f"canonicalized {n_canon[0]} generated names, merged {n_canon[1]} duplicate constants")
    print(f"onnx: {out} ({out.stat().st_size / 1e6:.2f} MB)")
    print("ops:", json.dumps(hist))

    # Still written, though the head now ships inside the graph as `cal_weight`: saccade.js
    # exposes these as the opt-in CAL_HEAD, for reproducing how 1.0.0 weighted its rows.
    kernel, bias = [np.asarray(w) for w in cal.get_weights()]
    write_json(GEN_DIR / "cal_weights.json", {
        "kernel": [float(v) for v in kernel[:, 0]],
        "bias": float(bias[0]),
    })

    write_json(GEN_DIR / "export_manifest.json", {
        "checkpoint": {
            "entity": ENTITY,
            "project": PROJECT,
            "run": RUN,
            "file": weights.name,
        },
        "model": {
            "file": "models/eye_embedding.onnx",
            "input": {"name": IN_NAME, "shape": list(IN_SHAPE), "dtype": "float32"},
            "output": {"name": OUT_NAME, "shape": [1, int(kernel.shape[0])],
                       "dtype": "float32"},
            # The per-image calibration weight, in [0, 1]. A model without one omits this key
            # and saccade.js fits unweighted; it is not an error, just less information.
            "weight_output": {"name": WEIGHT_NAME, "shape": [1, 1], "dtype": "float32",
                              "range": "sigmoid_0_1", "granularity": "per_image"},
            "input_range": "uint8_0_255",
            "embedding_dim": int(kernel.shape[0]),
            "opset": OPSET,
            "has_einsum": "Einsum" in hist,
            "domains": domains,
            "op_histogram": hist,
        },
        "ridge": {
            "lambda_default": 1.0,
            "lambda_fewpoint": 3.0,
            "center_targets": True,
            "center": 0.5,
        },
        "preprocess": {
            "pad": 0.01,
            "landmarks": {"left": 226, "right": 446, "top": [27, 257],
                          "bottom": [23, 253]},
            "gray": "bt601",
            "resize": {"method": "cv2_inter_linear", "width": 144, "height": 36},
            "clahe": {"clip": 2.0, "tiles": [8, 8]},
            "order": ["crop", "gray", "resize", "clahe"],
            "note": ("Browser landmarks come from the MediaPipe Tasks FaceLandmarker; "
                     "training used the legacy mp.solutions FaceMesh (refine_landmarks). "
                     "Indices match but coordinates differ slightly."),
        },
        "versions": {
            "keras": keras.__version__,
            "tensorflow": tf.__version__,
            "tf2onnx": tf2onnx.__version__,
            "onnx": onnx.__version__,
        },
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })
    print(f"wrote {GEN_DIR / 'cal_weights.json'}")
    print(f"wrote {GEN_DIR / 'export_manifest.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fuse", action="store_true",
                    help="skip the LayerNormalization/Gelu graph surgery")
    main(fuse=not ap.parse_args().no_fuse)
