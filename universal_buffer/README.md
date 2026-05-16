# universal_buffer

A header-only CUDA library for RAII-based GPU memory management and suballocation.

## Why UniversalDeviceBuffer?

**1. Runtime `cudaMalloc` introduces latency spikes**

`cudaMalloc` and `cudaFree` are not free operations — they can block for several milliseconds, and the latency is unpredictable. In latency-sensitive systems (e.g. online inference), calling `cudaMalloc` on the hot path can cause request-level spikes. The solution is to pre-allocate a large buffer at startup and suballocate from it at runtime, eliminating driver-level allocation overhead entirely. See `test_cuda_malloc_latency` for a benchmark that demonstrates this effect.

**2. Memory can be shared across modules that are never active simultaneously**

Consider a system where module A needs 100 MB and module B needs 80 MB. Naively pre-allocating both wastes 180 MB. But if A and B are never active at the same time, a single 100 MB buffer can be shared between them — A borrows it when needed, releases it when done, and B does the same. `UniversalDeviceBuffer` makes this pattern safe and automatic via RAII: a module calls `getBuffer` to claim a slice, and the slice is returned to the pool as soon as it goes out of scope.

## Components

### `cuda_malloc_raii.cuh`

RAII wrappers around `cudaMalloc` / `cudaMallocHost` using `shared_ptr` for reference-counted lifetime management. Copyable — the last copy to go out of scope frees the memory.

**Classes:**
- `CudaArray<T>` — base class (non-instantiable directly)
- `CudaDeviceArray<T>` — device memory
- `CudaHostArray<T>` — pinned host memory

**Constructors:**
```cpp
// Allocating — owns memory, frees on last copy destruction
CudaDeviceArray<float>(size, "name");

// Wrapping — borrows an external pointer, never frees
CudaDeviceArray<float>(ptr, size, "name");

// Wrapping with release signal — sets *isReleased = true on last copy destruction
CudaDeviceArray<float>(ptr, size, "name", isReleased);
```

### `universal_buffer.cuh`

A memory pool that pre-allocates a single device buffer and suballocates aligned slices from it. Slices are automatically returned to the pool when they go out of scope.

```cpp
UniversalDeviceBuffer buf(totalSizeInBytes, "name");

{
    CudaDeviceArray<char> slice = buf.getBuffer(sliceSizeInBytes);
    // use slice...
} // slice goes out of scope — memory is automatically reclaimed

buf.getTotalBytes(); // total size of the buffer
buf.getFreeBytes();  // currently available bytes
```

**Notes:**
- All allocations are aligned to 256 bytes for optimal CUDA memory access (see [CUDA Data Alignment](https://leimao.github.io/blog/CUDA-Data-Alignment/))
- `getBuffer` throws `std::runtime_error` if no contiguous free space of the requested size exists

## Build

```bash
./run.sh
```

Requires CUDA, CMake >= 3.18, and a C++17 compiler.

## Tests

| Executable | Covers |
|---|---|
| `test_cuda_malloc_raii` | `CudaDeviceArray`, `CudaHostArray` — allocation, copy semantics, external-pointer constructor |
| `test_universal_buffer` | `UniversalDeviceBuffer` — suballocation, auto-release, alignment, OOM |
