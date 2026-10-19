// Real CUDA D2H copy helper. Implemented in cuda_copy.cu and compiled with
// nvcc; declared here with a plain C++ signature (no CUDA types) so the
// rest of the benchmark (compiled with g++) can call it without needing the
// CUDA headers or compiler.
#pragma once

#include <vector>

// Allocates a device buffer of src.size() ints, copies src into it (H2D,
// untimed -- this models "the result already lives on the GPU"), then
// repeats `reps` timed D2H cudaMemcpy calls back into a fresh host vector.
// Returns the median (p50) and p95 latency of those D2H copies, in
// microseconds. Requires a CUDA-capable GPU at runtime; aborts with an
// error message if none is present.
struct GpuCopyStats
{
    double p50Us;
    double p95Us;
};

GpuCopyStats timedCudaD2HCopy(const std::vector<int>& src, int reps);
