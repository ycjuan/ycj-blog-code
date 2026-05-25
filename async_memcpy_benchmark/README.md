# async_memcpy_benchmark

Benchmarks four concurrent read/write strategies for a GPU vector index under mixed workloads: simultaneous scoring (reads) and upsert/delete/scalar-update (writes).

## Motivation

A GPU vector index stores embedding vectors in device memory. Concurrent writes (upserts, deletes, scalar updates) and reads (scoring) must be carefully coordinated to avoid serving stale or partially-written data. The naive approach serializes everything through a single mutex — simple but slow. This benchmark explores progressively more concurrent strategies and measures the tradeoff between correctness overhead and throughput.

## The Data Model

Each document has:
- An embedding vector (`bfloat16`, dimension 512) stored in a flat GPU array `m_data[rowIdx * embDim ... (rowIdx+1)*embDim]`
- A list of scalar values (`float32`, 32 per doc) stored in `m_d_scalars[rowIdx * numScalars + scalarIdx]`
- A dirty bit `m_d_dirty[rowIdx]` that marks a row as invisible to scorers

Documents are identified by a `long docId`. The index maps `docId -> rowIdx` (CPU-side hash map). Freed rowIdxs are recycled via a free-list.

Scoring takes a query embedding and a list of target `rowIdx` values, skips rows where `dirty==1`, and returns dot-product scores weighted by a selected scalar dimension (`targetScalarIdx`).

## The Four Strategies

### WorkerNaive

One global mutex serializes all operations (upserts, deletes, scalar updates, and scoring). Simple and correct but every operation blocks every other.

### WorkerOverwrite

Splits locking into two mutexes:
- `m_writeMutex`: protects the `docId -> rowIdx` map during resolution
- `m_readMutex`: protects dirty bits during score reads

Upsert writes in-place to the existing rowIdx. To prevent a scorer from reading a half-written embedding, the dirty bit is set to 1 before the scatter and cleared to 0 after. This means score and upsert can overlap — the scorer sees the row as dirty and skips it — but two concurrent upserts to the same row are still safe because the dirty bit prevents the scorer from seeing a torn write.

### WorkerCopyOnWriteEager

Instead of overwriting the existing row, upsert always allocates a **new** rowIdx for the document, scatters the new embedding there, then atomically flips dirty bits: mark the old row dirty (hidden) and mark the new row clean (visible).

This means the scorer never sees a partially-written embedding — either it reads the old clean row or the new clean row, never a half-written one. The flip is the only moment that requires holding `m_readMutex`.

Scalar data is kept both on CPU (in `m_docId2scalar`) and on GPU (`m_d_scalars`). `updateScalarData` scatters scalars to GPU immediately. `upsertDocs` carries the CPU-side scalars to the new rowIdx between the emb scatter and the dirty-bit flip, so the new row has up-to-date scalar values when it becomes visible.

### WorkerCopyOnWriteLazy

Same copy-on-write approach for embeddings. The difference is in how scalars are handled:

- `updateScalarData` stores scalars on CPU only — no GPU scatter at all.
- `upsertDocs` does not touch scalars; the new row's scalar slot in `m_d_scalars` is intentionally left stale.
- `score` syncs the CPU scalars for the target rows to GPU (scatter kernel) immediately before calling the score kernel. Both launches go on the same CUDA stream so ordering is guaranteed without an explicit sync between them.

This eliminates the scalar-carry step from the hot upsert path at the cost of doing a H2D scalar sync on every score call.

## Concurrency Summary

| Worker | Upsert vs Score | Upsert vs Upsert | Scalar update vs Score |
|---|---|---|---|
| Naive | Serialized | Serialized | Serialized |
| Overwrite | Overlapping (dirty bit) | Safe (same rowIdx) | Overlapping (dirty bit) |
| COW Eager | Overlapping (new rowIdx) | Safe (different rowIdxs) | Overlapping (GPU scalar always current) |
| COW Lazy | Overlapping (new rowIdx) | Safe (different rowIdxs) | Score syncs CPU→GPU per call |

## Benchmark

The benchmark (`test/test_async_memcpy_benchmark.cu`) runs each worker under a mixed workload for 3 seconds using three concurrent threads:

- **Score thread**: issues scoring requests at 2500 calls/sec, each scoring 10k randomly selected rows
- **Upsert+Delete thread**: upserts at 60k docs/sec in batches of 1000; half of each batch are existing docs (re-upserted) and half are new docs that are deleted immediately after to keep the index size stable
- **UpdateScalar thread**: updates scalars at 30k docs/sec in batches of 1000

The index is pre-loaded with 1M documents (out of a 2M capacity) before benchmarking begins.

Reported metrics: mean / p50 / p99 latency per operation, and observed throughput (calls/sec for score, docs/sec for upsert/delete/updateScalar).

## Build

```bash
mkdir -p async_memcpy_benchmark/build
cmake -S async_memcpy_benchmark -B async_memcpy_benchmark/build
cmake --build async_memcpy_benchmark/build
```

## Run

```bash
./async_memcpy_benchmark/run.sh
```

or directly:

```bash
./async_memcpy_benchmark/build/test_async_memcpy_benchmark
```
