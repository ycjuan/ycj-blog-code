# JAX Inference in C++

Demonstrates how to serve a JAX/Flax model in pure C++ with no Python runtime, across three backends: ONNX Runtime, IREE, and Pure CUDA.

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
| Model file | `model.onnx` | `model.vmfb` / `model_cuda.vmfb` | `weights/*.bin` |
| C++ library | ONNX Runtime | IREE runtime | cuBLAS only |
| GPU required | No (CPU EP) / Yes (CUDA EP) | No (llvm-cpu) / Yes (cuda) | Yes |
| Python at serve time | No | No | No |

## Step 1: Export the model

Requires Python 3.11+ for IREE CUDA support. Python 3.9 can export CPU-only artifacts.

```bash
# Python 3.11+ venv (for IREE CUDA vmfb + ONNX + CPU vmfb)
python3.11 -m venv venv && source venv/bin/activate
pip install "jax[cuda12]" flax onnx iree-base-compiler
python3 export.py
# Produces: model.onnx, model.vmfb (llvm-cpu), model_cuda.vmfb (cuda), weights/
```

## Step 2: Install dependencies

### ONNX Runtime

```bash
cd ~/external && wget https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-x64-gpu-1.26.0.tgz && tar -xzf onnxruntime-linux-x64-gpu-1.26.0.tgz
```

### IREE runtime with CUDA (build from source)

The pip `iree-base-compiler` ships no C static library, so the runtime must be built from source.
Build only the runtime (no compiler = no LLVM = ~10 min):

```bash
git clone --depth 1 --branch v3.11.0 https://github.com/iree-org/iree.git /tmp/iree-src
git -C /tmp/iree-src submodule update --init --depth 1 --recursive
mkdir /tmp/iree-build && cd /tmp/iree-build
cmake /tmp/iree-src \
    -DCMAKE_BUILD_TYPE=Release \
    -DIREE_BUILD_COMPILER=OFF \
    -DIREE_BUILD_TESTS=OFF \
    -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_BUILD_PYTHON_BINDINGS=OFF \
    -DIREE_HAL_DRIVER_CUDA=ON \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON
make -j$(nproc) iree_runtime_unified
# Copy artifacts to ~/external/iree-311-cuda/{lib,include}/
```

### Pure CUDA

No extra install — uses cuBLAS from the CUDA Toolkit.

## Step 3: Individual approaches

Each folder is a self-contained standalone example.

### Approach 1: ONNX Runtime

JAX has no official ONNX exporter, so `export.py` builds the ONNX graph manually using the `onnx` Python library.

```bash
cd onnxruntime && ./compile.sh && ./run.sh
```

### Approach 2: IREE

Uses `jax.export` to serialize the model as StableHLO, then `iree-compile` to AOT-compile it.
Two vmfb variants are produced: `model.vmfb` (llvm-cpu) and `model_cuda.vmfb` (cuda).

```bash
cd iree && ./compile.sh && ./run.sh
```

### Approach 3: Pure CUDA + cuBLAS

```bash
cd cuda && ./compile.sh && ./run.sh
```

## Step 4: Compare all backends

```bash
cd compare && ./compile.sh && ./run.sh
```

Expected output:

```
Initializing backends...
Checking correctness (num_docs=10000)...
[PASS] IREE (CPU)      vs ORT (CPU)
[PASS] ORT (GPU)       vs ORT (CPU)
[PASS] IREE (CUDA)     vs ORT (CPU)
[PASS] Pure CUDA       vs ORT (CPU)

Benchmarking (num_docs=10000, 3 warmup + 10 trials)...

  ONNX Runtime (CPU)         e2e:  12.16 ms
  IREE (CPU, local-sync)     e2e:  47.31 ms
  [A] H2D transfer              :    1.18 ms
  [C] D2H transfer              :    0.04 ms
  [A+C] total transfer          :    1.22 ms

  ONNX Runtime (GPU)         e2e:   2.13 ms
  IREE (CUDA)                e2e:  34.38 ms
  Pure CUDA                  e2e:   2.91 ms  kernel:   1.69 ms
```

## Benchmark

Config: Amazon Linux 2023, CUDA 12.9, T4 GPU.
Model: query\_dim=64, doc\_dim=128, hidden=[256, 128], num\_heads=2, num\_docs=10000.

The benchmark separately times three segments: **[A]** H2D transfer, **[B]** kernel, and **[C]** D2H transfer.

| Backend | e2e | kernel only |
|---|---|---|
| ONNX Runtime (CPU) | 12.16 ms | — |
| IREE (CPU, local-sync) | 47.31 ms | — |
| **ONNX Runtime (GPU)** | **2.13 ms** | — |
| IREE (CUDA) | 34.38 ms | — |
| Pure CUDA | 2.91 ms | 1.69 ms |

ONNX Runtime GPU and Pure CUDA are ~16x faster than IREE CUDA. IREE's CUDA codegen for this model produces unvectorized kernels — it launches one thread per output element rather than using cuBLAS-style tiling. ORT GPU uses cuDNN/cuBLAS internally and benefits from highly optimized GEMM implementations.

IREE CPU (`local-sync`, single-threaded) is 4x slower than ORT CPU because ORT uses its default multi-threaded execution pool.

## JAX → ONNX

JAX has no official ONNX exporter (the popular `jax2onnx` package requires Python 3.10+). `export.py` constructs the ONNX graph manually: each MLP layer becomes a `Gemm` node with weight initializers, with `Unsqueeze`/`Squeeze` to handle the query broadcast across docs.

## Run everything end-to-end

```bash
./run.sh
```

## Acknowledgements

The multi-layer MLP model in `export.py` is taken from a certain work done by my colleague [Benjamin Le](https://www.linkedin.com/in/benjaminhoanle/) with some simplifications.
