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

## Step 1: Install Python 3.11

IREE's CUDA compiler backend requires Python 3.11+. Install it via the system package manager (Python 3.9 can still export CPU-only artifacts):

```bash
# Amazon Linux 2023 / RHEL
sudo dnf install python3.11

# Ubuntu / Debian
# sudo apt install python3.11
```

## Step 2: Create a Python venv and export the model

```bash
python3.11 -m venv ~/external/venv311
source ~/external/venv311/bin/activate
pip install "jax[cuda12]" flax onnx iree-base-compiler

cd jax_inference_cpp
JAX_PLATFORMS=cpu python3 export.py
# Produces: model.onnx, model.vmfb (llvm-cpu), model_cuda.vmfb (cuda), weights/
```

## Step 3: Install C++ dependencies

### ONNX Runtime

Download the pre-built GPU package from the [ONNX Runtime releases page](https://github.com/microsoft/onnxruntime/releases):

```bash
mkdir -p ~/external && cd ~/external
wget https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-x64-gpu-1.26.0.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.26.0.tgz
# Produces: ~/external/onnxruntime-linux-x64-gpu-1.26.0/{include,lib}/
```

### IREE runtime with CUDA (build from source)

The pip `iree-base-compiler` package ships the compiler but no C static library for embedding. The runtime must be built from source. Building with `-DIREE_BUILD_COMPILER=OFF` skips LLVM compilation and finishes in ~10 minutes.

CMake 3.26+ is required (the system cmake may be older; install a newer one via pip):

```bash
pip install cmake   # installs cmake 4.x to ~/.local/bin/cmake
```

Clone IREE 3.11.0 and initialize the required submodules:

```bash
git clone --depth 1 --branch v3.11.0 https://github.com/iree-org/iree.git /tmp/iree-src
git -C /tmp/iree-src submodule update --init --depth 1 \
    third_party/flatcc \
    third_party/vulkan_headers \
    third_party/webgpu-headers \
    third_party/printf
```

Configure and build (runtime only, no compiler):

```bash
mkdir /tmp/iree-build && cd /tmp/iree-build
~/.local/bin/cmake /tmp/iree-src \
    -DCMAKE_BUILD_TYPE=Release \
    -DIREE_BUILD_COMPILER=OFF \
    -DIREE_BUILD_TESTS=OFF \
    -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_BUILD_PYTHON_BINDINGS=OFF \
    -DIREE_HAL_DRIVER_CUDA=ON \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON
make -j$(nproc) iree_runtime_unified
```

Install into `~/external/iree-311-cuda/`:

```bash
IREE_DIST=~/external/iree-311-cuda
mkdir -p $IREE_DIST/{lib,include}

cp /tmp/iree-build/runtime/src/iree/runtime/libiree_runtime_unified.a $IREE_DIST/lib/
cp /tmp/iree-build/build_tools/third_party/printf/libprintf_printf.a    $IREE_DIST/lib/
find /tmp/iree-build -name "libflatcc*.a" -exec cp {} $IREE_DIST/lib/ \;

cp -r /tmp/iree-src/runtime/src/iree $IREE_DIST/include/iree

# Optional: clean up build directories (~2 GB)
rm -rf /tmp/iree-build /tmp/iree-src
```

### Pure CUDA

No extra install — uses cuBLAS from the CUDA Toolkit.

## Step 4: Individual approaches

Each folder is a self-contained standalone example. `compile.sh` runs CMake and builds the binary; `run.sh` executes it.

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

## Step 5: Compare all backends

```bash
cd compare && ./compile.sh
```

The ONNX Runtime CUDA Execution Provider requires cuDNN. When JAX is installed via `pip install "jax[cuda12]"`, cuDNN is bundled under the venv's `nvidia/cudnn/lib/` directory. Add it to `LD_LIBRARY_PATH` before running:

```bash
CUDNN_LIB=~/external/venv311/lib/python3.11/site-packages/nvidia/cudnn/lib
LD_LIBRARY_PATH=~/external/onnxruntime-linux-x64-gpu-1.26.0/lib:${CUDNN_LIB}:$LD_LIBRARY_PATH \
    ./build/compare
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

  ONNX Runtime (CPU)         e2e:   7.46 ms
  IREE (CPU, local-sync)     e2e:  47.72 ms
  [A] H2D transfer              :    1.13 ms
  [C] D2H transfer              :    0.04 ms
  [A+C] total transfer          :    1.17 ms

  ONNX Runtime (GPU)         e2e:   1.93 ms  kernel:   0.76 ms
  IREE (CUDA)                e2e:  31.49 ms  kernel:  30.32 ms
  Pure CUDA                  e2e:   2.23 ms  kernel:   1.06 ms
```

## Benchmark

Config: Amazon Linux 2023, CUDA 12.9, T4 GPU.
Model: query\_dim=64, doc\_dim=128, hidden=[256, 128], num\_heads=2, num\_docs=10000.

The benchmark separately times three segments: **[A]** H2D transfer, **[B]** kernel, and **[C]** D2H transfer.

| Backend | e2e | kernel only |
|---|---|---|
| ONNX Runtime (CPU) | 7.46 ms | — |
| IREE (CPU, local-sync) | 47.72 ms | — |
| **ONNX Runtime (GPU)** | **1.93 ms** | **0.76 ms** |
| IREE (CUDA) | 31.49 ms | 30.32 ms |
| Pure CUDA | 2.23 ms | 1.06 ms |

ONNX Runtime GPU and Pure CUDA are ~16x faster than IREE CUDA. IREE's CUDA codegen for this model produces unvectorized kernels — it launches one thread per output element rather than using cuBLAS-style tiling. ORT GPU routes through cuDNN/cuBLAS, which uses highly tuned GEMM implementations with tensor cores.

IREE CPU (`local-sync`, single-threaded) is 4x slower than ORT CPU because ORT uses its default multi-threaded execution pool.

IREE's strength is portability across exotic targets (TPUs, mobile NPUs, custom accelerators) where cuBLAS doesn't exist. For standard GPU GEMM workloads on NVIDIA hardware, it is not competitive with ORT.

## JAX → ONNX

JAX has no official ONNX exporter (the popular `jax2onnx` package requires Python 3.10+). `export.py` constructs the ONNX graph manually: each MLP layer becomes a `Gemm` node with weight initializers, with `Unsqueeze`/`Squeeze` to handle the query broadcast across docs.

## Run everything end-to-end

```bash
./run.sh
```

## Acknowledgements

The multi-layer MLP model in `export.py` is taken from a certain work done by my colleague [Benjamin Le](https://www.linkedin.com/in/benjaminhoanle/) with some simplifications.
