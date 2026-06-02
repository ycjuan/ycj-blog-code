# PyTorch Inference in C++

Demonstrates five approaches to serve a PyTorch model in pure C++ with no Python runtime.

## Model

A 2-tower MLP scorer (query + doc):
- Layer 1: separate projections for query and doc, additively fused with ReLU
- Hidden layers: linear + ReLU stack
- Output: linear → sigmoid, with `num_heads` scores per doc

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
[PASS] ONNX Runtime  vs TorchScript
[PASS] TensorRT      vs TorchScript
[PASS] Pure CUDA     vs TorchScript
[PASS] AOTInductor   vs TorchScript

Benchmarking (num_docs=10000, 3 warmup + 10 trials)...

  [A] H2D transfer              :   1.14 ms
  [C] D2H transfer              :   0.04 ms
  [A+C] total transfer          :   1.18 ms

  TorchScript     e2e:  10.75 ms
  ONNX Runtime    e2e:  12.63 ms
  TensorRT        e2e:   1.77 ms  kernel:   0.59 ms
  AOTInductor     e2e:   1.80 ms  kernel:   0.62 ms
  Pure CUDA       e2e:   2.62 ms  kernel:   1.44 ms
```

## Benchmark

Config: Amazon Linux 2023, CUDA 12.9, TensorRT 11, T4 GPU.
Model: query\_dim=64, doc\_dim=128, hidden=[256, 128], num\_heads=2, num\_docs=10000.

The benchmark separately times three segments: **[A]** copying query and doc embeddings from CPU to GPU (H2D), **[B]** the inference kernel itself, and **[C]** copying scores back from GPU to CPU (D2H). This makes the comparison fair — H2D/D2H costs are the same across all GPU backends, so kernel time isolates each approach's actual compute efficiency.

| Backend | e2e | kernel only |
|---|---|---|
| TorchScript | 10.75 ms | — |
| ONNX Runtime | 12.63 ms | — |
| AOTInductor | 1.80 ms | 0.62 ms |
| Pure CUDA | 2.62 ms | 1.44 ms |
| **TensorRT** | **1.77 ms** | **0.59 ms** |

H2D transfer (1.14 ms) and D2H transfer (0.04 ms) are shared across all GPU backends. The e2e column is kernel + transfer.

**TorchScript and ONNX Runtime cannot report kernel-only time.** They are CPU-side runtimes that manage their own internal GPU memory, so there is no API to inject pre-allocated device pointers and skip the H2D step — inference is always an end-to-end black box.

TensorRT and AOTInductor are both ~5x faster than TorchScript/ONNX Runtime because they compile GPU kernels ahead of time and auto-tune tiling for the exact input shape. Pure CUDA uses generic cuBLAS GEMMs without that auto-tuning.

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
