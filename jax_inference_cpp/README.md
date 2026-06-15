# JAX Inference in C++

Demonstrates three approaches to serve a JAX/Flax model in pure C++ with no Python runtime.

## Model

A 2-tower MLP scorer (query + doc), defined in Flax:
- Layer 1: separate projections for query and doc, additively fused with ReLU
- Hidden layers: linear + ReLU stack
- Output: linear → sigmoid, with `num_heads` scores per doc

## Dependency comparison

| | ONNX Runtime | IREE | Pure CUDA |
|---|---|---|---|
| Convert via | manual ONNX graph | `jax.export` → StableHLO | weight dump |
| Convert deps | `jax`, `flax`, `onnx` | `jax`, `flax`, `iree-base-compiler` | `jax`, `flax` |
| Model file | `model.onnx` | `model.vmfb` | `weights/*.bin` |
| C++ library | ONNX Runtime | IREE runtime | cuBLAS only |
| NVIDIA GPU required | No | No\* | Yes |
| Python at serve time | No | No | No |

\* IREE supports GPU backends, but JAX 0.4.x on Python 3.9 generates StableHLO that IREE 3.x's CUDA codegen cannot distribute. GPU requires JAX 3.10+ on Python 3.10+.

## Step 1: Export the model

```bash
pip install jax==0.4.30 flax onnx iree-base-compiler
python3 export.py
# Produces: model.onnx, model.vmfb, weights/
```

## Step 2: Install dependencies

### ONNX Runtime (ONNX Runtime and compare)

```bash
cd ~/external && wget https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-x64-gpu-1.26.0.tgz && tar -xzf onnxruntime-linux-x64-gpu-1.26.0.tgz
# Produces: ~/external/onnxruntime-linux-x64-gpu-1.26.0/
```

### IREE runtime (IREE and compare)

Download the IREE distribution (must match the `iree-base-compiler` version used to compile `model.vmfb`) from the [IREE releases page](https://github.com/iree-org/iree/releases).

```bash
# Example: extract to ~/external so that ~/external/include/iree/ and ~/external/lib/libiree_runtime_unified.a exist
```

### Pure CUDA (cuda and compare)

No extra install needed — uses cuBLAS from the CUDA Toolkit.

## Step 3: Individual approaches

Each folder is a self-contained standalone example. `compile.sh` runs CMake and builds the binary; `run.sh` executes it.

### Approach 1: ONNX Runtime

JAX does not have an official ONNX exporter, so `export.py` builds the ONNX graph manually using the `onnx` Python library, with trained weights as initializers.

```bash
cd onnxruntime && ./compile.sh && ./run.sh
```

### Approach 2: IREE

Uses `jax.export` to serialize the model as StableHLO (`.mlir`), then `iree-compile` to AOT-compile it to a platform-native `.vmfb` bytecode module. The C++ runtime loads the vmfb and drives inference through the IREE runtime C API.

```bash
cd iree && ./compile.sh && ./run.sh
```

### Approach 3: Pure CUDA + cuBLAS

```bash
cd cuda && ./compile.sh && ./run.sh
```

## Step 4: Run all three and assert they agree

```bash
cd compare && ./compile.sh && ./run.sh
```

Expected output (CPU-only machine):

```
[INFO] No CUDA GPU detected; skipping Pure CUDA backend.
Initializing backends...
Checking correctness (num_docs=10000)...
[PASS] IREE       vs ONNX Runtime

Benchmarking (num_docs=10000, 3 warmup + 10 trials)...

  ONNX Runtime    e2e:  49.22 ms
  IREE (CPU)      e2e: 931.38 ms
```

## Benchmark

Config: Amazon Linux 2023, CUDA 12.9, T4 GPU.
Model: query\_dim=64, doc\_dim=128, hidden=[256, 128], num\_heads=2, num\_docs=10000.

The benchmark separately times three segments: **[A]** copying query and doc embeddings from CPU to GPU (H2D), **[B]** the inference kernel itself, and **[C]** copying scores back from GPU to CPU (D2H).

| Backend | e2e | kernel only |
|---|---|---|
| ONNX Runtime (CPU) | 49.22 ms | — |
| IREE (CPU, local-sync) | 931.38 ms | — |
| Pure CUDA | — | — |

> **Note:** IREE here uses the `local-sync` driver (single-threaded) due to a threading incompatibility on this machine. The `local-task` driver (multi-threaded) would be significantly faster. ONNX Runtime uses its default thread pool.

## IREE GPU limitation

IREE supports CUDA and other GPU backends, but there is a version compatibility constraint: the StableHLO produced by JAX 0.4.x contains broadcast patterns that IREE 3.x's CUDA codegen cannot distribute (`'func.func' op failed to distribute`). This is resolved by using JAX 3.10+ with Python 3.10+.

## JAX → ONNX

JAX has no official ONNX exporter (the popular `jax2onnx` package requires Python 3.10+). `export.py` constructs the ONNX graph manually: each MLP layer becomes a `Gemm` node with weight initializers, with `Unsqueeze`/`Squeeze` to handle the query broadcast across docs.

## Run everything end-to-end

```bash
./run.sh
```

## Acknowledgements

The multi-layer MLP model in `export.py` is taken from a certain work done by my colleague [Benjamin Le](https://www.linkedin.com/in/benjaminhoanle/) with some simplifications.
