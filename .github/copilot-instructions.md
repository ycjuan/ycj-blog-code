# Copilot Instructions

This repo is a personal collection of standalone CUDA/C++ experiments backing blog posts at
https://ycjuan.github.io. It is not a single application — it is dozens of independent
top-level directories, each self-contained with its own build. There is no shared library,
no monorepo build system, and no CI.

## Repository structure

Each top-level directory (e.g. `gemm/`, `topk/`, `sddmm/`, `compressible_embeddings/`) is an
independent experiment/benchmark exploring one CUDA/systems topic. Directories do not depend
on each other. When working in one, treat it in isolation — don't assume shared headers or
build config across directories.

Two build styles are used, roughly split by project age/complexity:
- **Simple Makefile** projects (e.g. `gemm`, `sddmm`, `kernel_if_else_overhead`): a single
  `nvcc` invocation building one binary from a `main.cu`. Build with `make`, run the produced
  binary directly.
- **CMake** projects (e.g. `topk`, `compressible_embeddings`, `universal_buffer`,
  `concurrent_read_write_paradigm`, `thread_pool`): build a shared library (`lib<name>.so`)
  from `src/`/`include/`, plus a `test_<name>` executable from `test/`. These typically ship a
  `run.sh` that does a clean CMake build and immediately runs the test binary:
  ```bash
  ./run.sh        # cmake .. && make -j N && ./test_<name>
  ./run.sh -a     # force a clean rebuild (rm -rf build) where supported
  ```
  Always build in a `build/` subdirectory (`mkdir -p build && cd build && cmake .. && make`).
  `compile_commands.json` is exported for clangd/IntelliSense support.

There are no unit test frameworks beyond ad hoc `test_*.cu` executables that assert/print
results when run — "running the test binary" is the test suite for that project.

## Naming conventions (see `CODING_STYLE.md`)

- CUDA `__global__` kernel functions: prefix `kn_`, no `Kernel` suffix
  (`kn_scatter`, not `scatterKernel`).
- Device-memory pointers/arrays (GPU memory, including kernel parameters): prefix `d_`
  (`d_rowIdx`, `d_dirty`, `d_elements`).
- Vector variables, prefixed by dimensionality, using the singular element name:
  - `v_` for a 1D vector (`v_docId`, `v_rowIdx`, `v_scalar`)
  - `vv_` for a vector of vectors (`vv_embData`)

## Git workflow

- Always create new commits; never amend existing commits.
- When creating a new file, `git add` it immediately.
- Never use `git add -A` or `git add .` — add specific files by name.
- New branches must be prefixed `YYMMDD-` (e.g. `260601-my-feature`).

## Scope note

Per `README.md`, this is a personal repo shared for reference; issues and PRs from others are
not addressed.
