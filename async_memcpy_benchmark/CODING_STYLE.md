# Coding Style

## CUDA Kernel Naming

CUDA `__global__` kernel functions use a `kn_` prefix and no `Kernel` suffix.

- Good: `kn_scatter`, `kn_setDirty`, `kn_score`
- Bad: `scatterKernel`, `setDirtyKernel`, `scoreKernel`
