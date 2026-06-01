# PyTorch Inference in C++

Demonstrates four approaches to serve a PyTorch model in pure C++ with no Python runtime.

## Model

A 2-tower MLP scorer (query + doc):
- Layer 1: separate projections for query and doc, additively fused with ReLU
- Hidden layers: linear + ReLU stack
- Output: linear → sigmoid, with `num_heads` scores per doc

## Step 1: Export the model

```bash
pip install onnx
python3 export.py
# Produces: model.pt, model.onnx, weights/
```

## Step 2: Install dependencies

### LibTorch (TorchScript and compare)

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

```bash
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf install tensorrt
```

### Pure CUDA (cuda and compare)

No extra install needed — uses cuBLAS from the CUDA Toolkit.

## Step 3: Individual approaches

Each folder is a self-contained standalone example.

### Approach 1: TorchScript + LibTorch

```bash
cd torchscript && ./compile.sh && ./run.sh
```

### Approach 2: ONNX Runtime

```bash
cd onnxruntime && ./compile.sh && ./run.sh
```

### Approach 3: TensorRT (requires GPU)

```bash
cd tensorrt && ./compile.sh && ./run.sh
```

### Approach 4: Pure CUDA + cuBLAS

```bash
cd cuda && ./compile.sh && ./run.sh
```

## Step 4: Run all four and assert they agree

```bash
cd compare && ./compile.sh && ./run.sh
```

Expected output:

```
[PASS] ONNX Runtime vs TorchScript
[PASS] TensorRT     vs TorchScript
[PASS] Pure CUDA    vs TorchScript

Benchmarking (num_docs=10000, 3 warmup + 10 trials)...
  TorchScript  :    9.17 ms
  ONNX Runtime :    6.05 ms
  TensorRT     :    1.94 ms
  Pure CUDA    :    3.12 ms

All backends agree.
```

Benchmark config: Amazon Linux 2023, CUDA 12.9, TensorRT 11, T4 GPU.
Model: query\_dim=64, doc\_dim=128, hidden=[256, 128], num\_heads=2.

TensorRT is ~7x faster than TorchScript/ONNX Runtime because its engine compilation step tunes kernel tiling for the exact shape. Pure CUDA sits in between, using generic cuBLAS without TRT's auto-tuning.

## Acknowledgements

The multi-layer MLP model in `export.py` is taken from a certain work done by my colleague [Benjamin Le](https://www.linkedin.com/in/benjaminhoanle/) with some simplifications.
