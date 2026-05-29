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

```bash
cd ~/external && wget https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.8.0%2Bcu128.zip && unzip libtorch-*.zip
# Produces: ~/external/libtorch/
```

### ONNX Runtime (for ONNX Runtime approach)

```bash
cd ~/external && wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-gpu-cuda12-1.22.0.tgz && tar -xzf onnxruntime-*.tgz
# Produces: ~/external/onnxruntime-linux-x64-gpu-cuda12-1.22.0/
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
./compile.sh ~/external/libtorch
./build/inference
```

### Approach 2: ONNX Runtime

```bash
cd onnxruntime
./compile.sh ~/external/onnxruntime-linux-x64-gpu-cuda12-1.22.0
./build/inference
```

### Approach 3: TensorRT (requires GPU)

```bash
cd tensorrt
./compile.sh
./build/inference
```
