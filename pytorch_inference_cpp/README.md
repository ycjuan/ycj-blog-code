# PyTorch Inference in C++

Demonstrates five approaches to serve a PyTorch model in pure C++ with no Python runtime.

## Model

A 2-tower MLP scorer (query + doc), batched over R queries x N docs:
- Layer 1: separate projections for query and doc, additively fused with ReLU.
  The query projection is computed once per query and broadcast across docs;
  the doc projection is computed once per doc and broadcast across queries.
  `num_reqs == 1` is just a batch of size 1, not a separate code path.
- Hidden layers: linear + ReLU stack
- Output: linear → sigmoid, with `num_heads` scores per (query, doc) pair

## Dependency comparison

| | TorchScript | ONNX Runtime | TensorRT | AOTInductor | Pure CUDA |
|---|---|---|---|---|---|
| Convert via | `torch.jit.trace` | `torch.onnx.export` | `torch.onnx.export` | `torch.export` | weight dump |
| Convert deps | PyTorch | PyTorch + `onnx` | PyTorch + `onnx` | PyTorch | PyTorch |
| Model file | `model.pt` | `model.onnx` | `model.onnx` | `model.pt2` | `weights/*.bin` |
| C++ library | LibTorch | ONNX Runtime | TensorRT + CUDA | LibTorch | cuBLAS only |
| NVIDIA GPU required | No | No | Yes | Yes | Yes |
| PyTorch at serve time | No | No | No | No | No |

## Step 1: Export the model

```bash
pip install torch onnx
python3 export.py
# Produces: model.pt, model.onnx, model.pt2, weights/
```

## Step 2: Install dependencies

### LibTorch (TorchScript, AOTInductor, and compare)

```bash
cd ~/external && wget https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.8.0%2Bcu128.zip && unzip libtorch-*.zip
# Produces: ~/external/libtorch/
```

### ONNX Runtime (ONNX Runtime and compare)

```bash
cd ~/external && wget https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-x64-gpu-1.26.0.tgz && tar -xzf onnxruntime-linux-x64-gpu-1.26.0.tgz
# Produces: ~/external/onnxruntime-linux-x64-gpu-1.26.0/
```

### TensorRT (TensorRT and compare — requires GPU)

The command below works on Amazon Linux 2023. For other distributions, refer to the [NVIDIA TensorRT installation guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).

```bash
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf install tensorrt
```

### Pure CUDA (cuda and compare)

No extra install needed — uses cuBLAS from the CUDA Toolkit.

## Step 3: Individual approaches

Each folder is a self-contained standalone example. `compile.sh` runs CMake and builds the binary; `run.sh` executes it.

### Approach 1: TorchScript + LibTorch

> **Note:** TorchScript is largely deprecated. PyTorch 2.x introduced `torch.export` as the modern replacement, and the TorchScript-based ONNX exporter is being superseded by a `torch.export`-based one starting in PyTorch 2.9. TorchScript still works but is in maintenance mode. See [PyTorch docs](https://docs.pytorch.org/docs/2.12/notes/cpu_threading_torchscript_inference.html) for more details.

```bash
cd torchscript && ./compile.sh && ./run.sh
```

### Approach 2: ONNX Runtime

```bash
cd onnxruntime && ./compile.sh && ./run.sh
```

### Approach 3: TensorRT (requires GPU)

TensorRT does not use the ONNX Runtime library, but it does read `model.onnx`
directly — `nvonnxparser` parses it at startup to compile a GPU-optimized engine.

```bash
cd tensorrt && ./compile.sh && ./run.sh
```

### Approach 4: AOTInductor (requires GPU)

Uses `torch.export` + `torch._inductor.aoti_compile_and_package` to compile the model into a `.pt2` package with Inductor's kernel auto-tuning. Loaded in C++ via `AOTIModelPackageLoader`, which is part of LibTorch — no extra dependency beyond LibTorch.

```bash
cd aotinductor && ./compile.sh && ./run.sh
```

### Approach 5: Pure CUDA + cuBLAS

```bash
cd cuda && ./compile.sh && ./run.sh
```

## Step 4: Run all five and assert they agree

```bash
cd compare && ./compile.sh && ./run.sh
```

Expected output:

```
Checking correctness (num_reqs=32, num_docs=10000)...
[PASS] ONNX Runtime  vs TorchScript
[PASS] TensorRT      vs TorchScript
[PASS] AOTInductor   vs TorchScript
[PASS] Pure CUDA     vs TorchScript

Benchmarking (num_reqs=32, num_docs=10000, 3 warmup + 10 trials)...

  [A] H2D transfer              :   1.14 ms
  [C] D2H transfer              :   0.66 ms
  [A+C] total transfer          :   1.80 ms

  TorchScript     e2e: 485.57 ms
  ONNX Runtime    e2e: 171.80 ms
  TensorRT        e2e:  10.96 ms  kernel:   9.16 ms
  AOTInductor     e2e:  11.90 ms  kernel:  10.11 ms
  Pure CUDA       e2e:  22.31 ms  kernel:  20.52 ms

  Batched (32 reqs/call) vs one-req-at-a-time (32 calls):
  TensorRT        batched:    9.16 ms   1-by-1:   15.70 ms   speedup:  1.71x  (kernel-only)
  AOTInductor     batched:   10.11 ms   1-by-1:   17.49 ms   speedup:  1.73x  (kernel-only)
  Pure CUDA       batched:   20.52 ms   1-by-1:   42.64 ms   speedup:  2.08x  (kernel-only)
  TorchScript     batched:  485.57 ms   1-by-1:  341.30 ms   speedup:  0.70x  (e2e)
  ONNX Runtime    batched:  171.80 ms   1-by-1:  266.58 ms   speedup:  1.55x  (e2e)
```

## Benchmark

Config: Amazon Linux 2023, CUDA 12.9, TensorRT 11, T4 GPU.
Model: query\_dim=64, doc\_dim=128, hidden=[256, 128], num\_heads=2, num\_reqs=32, num\_docs=10000.

The benchmark separately times three segments: **[A]** copying query and doc embeddings from CPU to GPU (H2D), **[B]** the inference kernel itself, and **[C]** copying scores back from GPU to CPU (D2H). This makes the comparison fair — H2D/D2H costs are the same across all GPU backends, so kernel time isolates each approach's actual compute efficiency.

| Backend | e2e | kernel only |
|---|---|---|
| TorchScript | 485.57 ms | — |
| ONNX Runtime | 171.80 ms | — |
| **TensorRT** | **10.96 ms** | **9.16 ms** |
| AOTInductor | 11.90 ms | 10.11 ms |
| Pure CUDA | 22.31 ms | 20.52 ms |

H2D transfer (1.14 ms) and D2H transfer (0.66 ms) are shared across all GPU backends. The e2e column is kernel + transfer.

**TorchScript and ONNX Runtime cannot report kernel-only time.** They are CPU-side runtimes that manage their own internal GPU memory, so there is no API to inject pre-allocated device pointers and skip the H2D step — inference is always an end-to-end black box.

TensorRT and AOTInductor are both ~2x faster than Pure CUDA, and far faster than TorchScript/ONNX Runtime, because they compile GPU kernels ahead of time and auto-tune tiling for the exact input shape. Pure CUDA uses generic cuBLAS GEMMs without that auto-tuning.

### Batching 32 reqs vs scoring them one at a time

Scoring 32 queries in a single batched call is faster than looping over `infer`/`infer_device` 32 times with `num_reqs=1`, because the doc-side GEMMs (the dominant cost at `num_docs=10000`) and the doc-to-GPU transfer are shared across all 32 queries instead of repeated per query:

| Backend | batched (32 reqs/call) | 1-by-1 (32 calls) | speedup |
|---|---|---|---|
| TensorRT (kernel) | 9.16 ms | 15.70 ms | 1.71x |
| AOTInductor (kernel) | 10.11 ms | 17.49 ms | 1.73x |
| Pure CUDA (kernel) | 20.52 ms | 42.64 ms | 2.08x |
| ONNX Runtime (e2e) | 171.80 ms | 266.58 ms | 1.55x |
| TorchScript (e2e) | 485.57 ms | 341.30 ms | 0.70x |

TorchScript is the outlier: batching made it *slower*, likely due to its eager CPU interpreter allocating one large `[32, 10000, 2]` intermediate activation tensor per hidden layer instead of 32 smaller ones, which hurts cache locality more than it saves on call overhead. The other four backends all benefit from batching, with the hand-written CUDA kernel benefiting the most since it has no per-call setup cost to amortize differently.

## TensorRT vs AOTInductor

- **TensorRT**: slightly faster; supports all ONNX-compatible models, not just PyTorch
- **AOTInductor**: PyTorch-native, no dependency beyond LibTorch

## Run everything end-to-end

To export, compile, and run all five approaches in one shot:

```bash
./run.sh
```

## Acknowledgements

The multi-layer MLP model in `export.py` is taken from a certain work done by my colleague [Benjamin Le](https://www.linkedin.com/in/benjaminhoanle/) with some simplifications.
