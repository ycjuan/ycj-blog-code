# Embedding Quantization

A CUDA benchmark comparing methods for approximate nearest neighbor search via embedding quantization. Each method compresses document embeddings offline and reconstructs them at query time, trading reconstruction accuracy for memory bandwidth.

## Methods

### Baseline
Stores full-precision (`bfloat16`) embeddings and copies them to the GPU at query time (H2D) or reads them directly from device memory (D2D). This is the reference for latency and zero reconstruction error.

### Residual Quantization (ResQuant)
Assigns each document to its nearest centroid and quantizes the residual (embedding − centroid) using a uniform scalar quantizer, packing 2 bits per dimension into `uint64_t` words.

### TurboQuant ([paper](https://arxiv.org/abs/2504.19874))
Applies a Randomized Hadamard Transform (RHT) to the residual before quantizing with a Lloyd-Max scalar quantizer tuned for Gaussian inputs. The RHT spreads quantization error uniformly across all dimensions, and Lloyd-Max centroids minimize MSE for the resulting near-Gaussian distribution — together they reduce RMSE by ~45% relative to ResQuant.

The WHT kernel uses a three-tier butterfly strategy to minimize shared memory:
- **Tier 1** (strides < `elemsPerThread`): pure register arithmetic
- **Tier 2** (strides `elemsPerThread`..`31×elemsPerThread`): `__shfl_xor_sync` for intra-warp exchange
- **Tier 3** (strides ≥ `32×elemsPerThread`): single shared memory buffer (half of ping-pong, doubling occupancy)

## Data Layout

All arrays use row-major layout. Quantized residuals are bit-packed: with `numBitsPerDim=2` and `RQ_T=uint64_t`, each 64-bit word holds 32 dimensions. The centroid value table is stored flat (`numCentroids × embDim`) for coalesced access.

## Results

On an NVIDIA T4 (SM75), with `numDocs=100K`, `numToScore=10K`, `embDim=1024`, `numBitsPerDim=2`, `numCentroids=1024`:

| Method             | RMSE     | Latency  |
|--------------------|----------|----------|
| Baseline H2D       | 0.000    | 3.28 ms  |
| Baseline D2D       | 0.000    | 0.64 ms  |
| ResQuant H2D       | 0.107    | 1.14 ms  |
| ResQuant D2D       | 0.107    | 1.00 ms  |
| TurboQuant H2D     | 0.059    | 3.36 ms  |
| TurboQuant D2D     | 0.059    | 2.52 ms  |

RMSE is normalized per-document: `mean over docs of (‖reconstructed − original‖₂ / ‖original‖₂)`.

## Build

Requires CUDA, CMake ≥ 3.18, and OpenMP.

```bash
bash run.sh
```

Or manually:

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./test_embedding_quantization_unit_tests
./test_embedding_quantization
```

Targets SM75 (T4), SM80 (A100), SM86, SM89 (RTX 4090), SM90 (H100).
