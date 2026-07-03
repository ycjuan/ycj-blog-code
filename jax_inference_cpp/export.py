import os

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh

# ---------------------------------------------------------------------------
# Model definition (same architecture as pytorch_inference_cpp/export.py)
# ---------------------------------------------------------------------------
QUERY_DIM    = 64
DOC_DIM      = 128
HIDDEN_SIZES = [256, 128]
NUM_HEADS    = 2


class MlpScorer(nn.Module):
    hidden_sizes: tuple
    num_heads: int

    @nn.compact
    def __call__(self, query, docs):
        """
        query : [Dq]      — single query embedding
        docs  : [N, Dd]   — N doc embeddings
        returns [N, num_heads]
        """
        # Layer 1: separate projections for query and doc, bias on query side
        query_proj = nn.Dense(self.hidden_sizes[0], use_bias=True)(query)   # [H1]
        h = nn.Dense(self.hidden_sizes[0], use_bias=False)(docs) + query_proj  # [N, H1]
        h = nn.relu(h)

        # Hidden layers
        for size in self.hidden_sizes[1:]:
            h = nn.Dense(size)(h)
            h = nn.relu(h)

        return nn.sigmoid(nn.Dense(self.num_heads)(h))   # [N, num_heads]


model = MlpScorer(hidden_sizes=tuple(HIDDEN_SIZES), num_heads=NUM_HEADS)

key = jax.random.PRNGKey(0)
dummy_query = jax.random.normal(key, (QUERY_DIM,))
dummy_docs  = jax.random.normal(key, (5, DOC_DIM))

params = model.init(key, dummy_query, dummy_docs)

with jax.disable_jit():
    out = model.apply(params, dummy_query, dummy_docs)
print(f"Output shape: {out.shape}")   # expect (5, 2)
print(f"Output:\n{out}")

# ---------------------------------------------------------------------------
# Export A: ONNX (manual graph construction via the onnx library)
# ---------------------------------------------------------------------------
# Flatten params into named numpy arrays matching the C++ weight layout.
# Flax dense layers store weights as (in_dim, out_dim) — transpose to (out_dim, in_dim)
# to match PyTorch's convention used by the existing CUDA backend.

def p(layer, param):
    return np.array(params["params"][layer][param])

# Flax stores kernels as [in_dim, out_dim]; transpose to [out_dim, in_dim]
w1_query = p("Dense_0", "kernel").T          # [H1, Dq]
b1       = p("Dense_0", "bias")              # [H1]
w1_doc   = p("Dense_1", "kernel").T          # [H1, Dd]

# Dense_2 ... Dense_{1+len(HIDDEN_SIZES)-1} are the inter-hidden layers
num_hidden_layers = len(HIDDEN_SIZES) - 1    # layers between first and output
hidden_w = []
hidden_b = []
for i in range(num_hidden_layers):
    prefix = f"Dense_{2 + i}"
    hidden_w.append(p(prefix, "kernel").T)   # [H_next, H_prev]
    hidden_b.append(p(prefix, "bias"))

out_layer = f"Dense_{2 + num_hidden_layers}"
w_out = p(out_layer, "kernel").T             # [num_heads, H_last]
b_out = p(out_layer, "bias")                 # [num_heads]

def make_initializer(name, arr):
    return onh.from_array(arr.astype(np.float32), name=name)

nodes = []
inits = []

def add_init(name, arr):
    inits.append(make_initializer(name, arr))

add_init("W1_query", w1_query)   # [H1, Dq]
add_init("b1",       b1)
add_init("W1_doc",   w1_doc)     # [H1, Dd]

# query_proj = Gemm(query_2d, W1_query, b1, transB=1) where query is unsqueezed to [1,Dq]
nodes.append(oh.make_node("Unsqueeze", ["query", "axis_0"], ["query_2d"]))
nodes.append(oh.make_node("Gemm", ["query_2d", "W1_query", "b1"], ["query_proj_2d"],
                           transB=1))
nodes.append(oh.make_node("Squeeze", ["query_proj_2d", "axis_0"], ["query_proj"]))

# docs_proj = Gemm(docs, W1_doc, transB=1) — no bias (fused from query side)
nodes.append(oh.make_node("Gemm", ["docs", "W1_doc"], ["docs_proj"],
                           transB=1, beta=0.0))

# h1 = Relu(docs_proj + query_proj)  — Add broadcasts query_proj over N
nodes.append(oh.make_node("Add", ["docs_proj", "query_proj"], ["h1_pre"]))
nodes.append(oh.make_node("Relu", ["h1_pre"], ["h1"]))

cur = "h1"
for i, (wh, bh) in enumerate(zip(hidden_w, hidden_b)):
    wname, bname, out_name = f"W_hid_{i}", f"b_hid_{i}", f"h{i+2}"
    add_init(wname, wh)
    add_init(bname, bh)
    nodes.append(oh.make_node("Gemm", [cur, wname, bname], [f"{out_name}_pre"], transB=1))
    nodes.append(oh.make_node("Relu", [f"{out_name}_pre"], [out_name]))
    cur = out_name

add_init("W_out", w_out)
add_init("b_out", b_out)
nodes.append(oh.make_node("Gemm", [cur, "W_out", "b_out"], ["scores_pre"], transB=1))
nodes.append(oh.make_node("Sigmoid", ["scores_pre"], ["scores"]))

# axis_0 scalar for Unsqueeze/Squeeze
inits.append(onh.from_array(np.array([0], dtype=np.int64), name="axis_0"))

graph = oh.make_graph(
    nodes,
    "mlp_scorer",
    [
        oh.make_tensor_value_info("query",  onnx.TensorProto.FLOAT, [QUERY_DIM]),
        oh.make_tensor_value_info("docs",   onnx.TensorProto.FLOAT, [None, DOC_DIM]),
    ],
    [
        oh.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, [None, NUM_HEADS]),
    ],
    initializer=inits,
)

onnx_model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, "model.onnx")
print("Saved model.onnx")

# ---------------------------------------------------------------------------
# Export B: IREE vmfb (JAX → StableHLO → iree-compile)
# ---------------------------------------------------------------------------
import jax.export as jax_export

@jax.jit
def inference(query, docs):
    return model.apply(params, query, docs)

# Export with dynamic num_docs dimension
dummy_args = (jnp.ones((QUERY_DIM,)), jnp.ones((5, DOC_DIM)))
query_spec, docs_spec = jax_export.symbolic_args_specs(dummy_args, ["_", "n, _"])

exported = jax_export.export(inference)(query_spec, docs_spec)
stablehlo_text = exported.mlir_module()

import subprocess, tempfile, shutil

def iree_compile_mlir(mlir_text, output_path, extra_flags):
    """Compile StableHLO text to a vmfb via iree-compile."""
    # Prefer venv311's iree-compile (3.11.0, bytecode v17) even when not activated,
    # because the bundled IREE runtime only accepts v17.
    venv311_compiler = os.path.expanduser("~/venv311/bin/iree-compile")
    iree_compile = (
        venv311_compiler if os.path.exists(venv311_compiler)
        else shutil.which("iree-compile")
        or os.path.expanduser("~/external/bin/iree-compile")
    )
    if not iree_compile or not os.path.exists(iree_compile):
        raise FileNotFoundError(f"iree-compile not found; looked for: {iree_compile}")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as tmp:
            tmp.write(mlir_text)
            tmp_path = tmp.name
        result = subprocess.run(
            [iree_compile, tmp_path, "--iree-input-type=stablehlo",
             "--iree-opt-const-eval=false", "-o", output_path] + extra_flags,
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"iree-compile failed:\n{result.stderr}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# CPU vmfb via llvm-cpu backend (works with any Python version)
iree_compile_mlir(stablehlo_text, "model.vmfb", [
    "--iree-hal-target-backends=llvm-cpu",
    "--iree-llvmcpu-target-cpu=host",
])
with open("model.vmfb", "rb") as f:
    print(f"Saved model.vmfb ({len(f.read())} bytes)")

# CUDA vmfb via cuda backend (requires iree-compile with CUDA support, e.g. from venv311)
try:
    iree_compile_mlir(stablehlo_text, "model_cuda.vmfb", [
        "--iree-hal-target-backends=cuda",
        "--iree-hal-target-device=cuda",
        "--iree-cuda-target=sm_75",
    ])
    with open("model_cuda.vmfb", "rb") as f:
        print(f"Saved model_cuda.vmfb ({len(f.read())} bytes)")
except RuntimeError as e:
    print(f"[SKIP] model_cuda.vmfb: {str(e)[:120]}")

# ---------------------------------------------------------------------------
# Export C: raw weights (same binary format as pytorch_inference_cpp)
# ---------------------------------------------------------------------------
weights_dir = "weights"
os.makedirs(weights_dir, exist_ok=True)

def save_bin(arr, name):
    path = os.path.join(weights_dir, name)
    np.array(arr, dtype=np.float32).tofile(path)
    print(f"  Saved {path}  shape={list(arr.shape)}")

print("Saving raw weights:")
save_bin(w1_query, "w1_query.bin")   # [H1, Dq]
save_bin(b1,       "b1.bin")         # [H1]
save_bin(w1_doc,   "w1_doc.bin")     # [H1, Dd]
for i, (wh, bh) in enumerate(zip(hidden_w, hidden_b)):
    save_bin(wh, f"w_hidden_{i}.bin")
    save_bin(bh, f"b_hidden_{i}.bin")
save_bin(w_out, "w_out.bin")
save_bin(b_out, "b_out.bin")

import json
config = {
    "query_dim":    QUERY_DIM,
    "doc_dim":      DOC_DIM,
    "hidden_sizes": HIDDEN_SIZES,
    "num_heads":    NUM_HEADS,
}
with open(os.path.join(weights_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
print(f"  Saved {weights_dir}/config.json")
