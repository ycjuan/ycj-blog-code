# Coding Style

## CUDA Kernel Naming

CUDA `__global__` kernel functions use a `kn_` prefix and no `Kernel` suffix.

- Good: `kn_scatter`, `kn_setDirty`, `kn_score`
- Bad: `scatterKernel`, `setDirtyKernel`, `scoreKernel`

## Vector Variable Naming

Prefix vector variables by dimensionality. Use the singular form of the element name (no plural).

- `v_` — 1D vector: `v_docId`, `v_rowIdx`, `v_scalar`
- `v2_` — 2D vector (vector of vectors): `v2_embData`
