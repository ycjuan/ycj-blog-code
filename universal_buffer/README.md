# universal_buffer

A header-only CUDA library for RAII-based GPU memory management and suballocation.

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
