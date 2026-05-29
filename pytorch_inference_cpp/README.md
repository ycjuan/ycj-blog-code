# PyTorch Inference in C++

Demonstrates three approaches to serve a PyTorch model in pure C++ with no Python runtime.

## Model

A simple 2-layer MLP: input(4) → hidden(8) → output(2), with random weights.

## Step 1: Export the model

```bash
pip install onnx
python3 export.py
# Produces: model.pt (TorchScript), model.onnx (ONNX Runtime + TensorRT)
```

## Step 2: Install dependencies

### LibTorch (for TorchScript approach)

Download from https://pytorch.org/get-started/locally/ — select Stable, Linux, LibTorch, C++/Java, CUDA 12.8.

```bash
cd ~ && unzip libtorch-*.zip
# Produces: ~/libtorch/
```

### ONNX Runtime (for ONNX Runtime approach)

Download the CUDA 12 variant from https://github.com/microsoft/onnxruntime/releases — pick `onnxruntime-linux-x64-gpu-cuda12-*.tgz`.

```bash
cd ~ && tar -xzf onnxruntime-linux-x64-gpu-cuda12-*.tgz
# Produces: ~/onnxruntime-linux-x64-gpu-cuda12-<version>/
```

### TensorRT (for TensorRT approach)

Add the NVIDIA CUDA repo (Amazon Linux 2023 / RHEL9):

```bash
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf install tensorrt
```

## Step 3: Compile and run

### Approach 1: TorchScript + LibTorch

```bash
cd torchscript
./compile.sh ~/libtorch
./build/inference
```

### Approach 2: ONNX Runtime

```bash
cd onnxruntime
./compile.sh ~/onnxruntime-linux-x64-gpu-cuda12-<version>
./build/inference
```

### Approach 3: TensorRT

```bash
cd tensorrt
./compile.sh
./build/inference
```
