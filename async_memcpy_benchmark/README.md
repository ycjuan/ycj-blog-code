# async_memcpy_benchmark

Benchmarks four concurrent read/write strategies for a GPU vector index under mixed workloads: simultaneous scoring (reads) and upsert/delete/scalar-update (writes).

## Motivation

A GPU vector index stores embedding vectors in device memory. Concurrent writes (upserts, deletes, scalar updates) and reads (scoring) must be carefully coordinated to avoid serving stale or partially-written data. The naive approach serializes everything through a single mutex — simple but slow. This benchmark explores progressively more concurrent strategies and measures the tradeoff between correctness overhead and throughput.

## The Data Model

All GPU arrays use **row-major** layout. Each document has:
- An embedding vector (`bfloat16`, dimension 512) stored in a flat GPU array `m_data[rowIdx * embDim ... (rowIdx+1)*embDim]`
- A list of scalar values (`float32`, 32 per doc) stored in `m_d_scalars[rowIdx * numScalars + scalarIdx]`
- A dirty bit `m_d_dirty[rowIdx]` (`DirtyBit::CLEAN` or `DirtyBit::DIRTY`) that marks a row as invisible to scorers

Documents are identified by a `long docId`. The index maps `docId -> rowIdx` (CPU-side hash map). Freed rowIdxs are recycled via a free-list.

Scoring takes a query embedding and a list of target `rowIdx` values, skips rows where `dirty==1`, and returns dot-product scores weighted by a selected scalar dimension (`targetScalarIdx`).

There are two types of writes:
- **Primary update (upsert):** replaces a document's embedding. This is the expensive operation — 512 bfloat16 values must be transferred to the GPU.
- **Secondary update (scalar):** updates a document's scalar weights without touching the embedding. Much cheaper, but must still be coordinated with concurrent scorers reading the same scalar slots.

## Design Challenges

Getting concurrent reads and writes correct requires navigating four tensions:

1. **Race condition** — A scorer and a writer touching the same GPU memory location from different CUDA streams have no ordering guarantee. Without explicit coordination, the scorer can read a partially-written value.

2. **Concurrency** — If every operation must wait for every other, throughput suffers. The goal is to let scores and writes overlap as much as possible.

3. **Doc visibility** — During an update, a document may need to be temporarily hidden from scorers (dirty=1) to prevent torn reads. Any score request that arrives during this window gets results as if the document does not exist.

4. **docId↔rowIdx map mutation** — Some strategies (copy-on-write) allocate a new rowIdx on every upsert and free the old one. This keeps the embedding write invisible until committed, but it means the map is constantly changing, which adds overhead and complexity.

## The Four Strategies

### WorkerNaive

One global mutex serializes all operations (upserts, deletes, scalar updates, and scoring). Simple and correct but every operation blocks every other.

### WorkerOverwrite

Splits locking into two mutexes:
- `m_writeMutex`: protects the `docId -> rowIdx` map during resolution
- `m_readMutex`: protects dirty bits during score reads

Upsert writes in-place to the existing rowIdx. To prevent a scorer from reading a half-written embedding, the dirty bit is set to 1 before the scatter and cleared to 0 after. This means score and upsert can overlap — the scorer sees the row as dirty and skips it — but two concurrent upserts to the same row are still safe because the dirty bit prevents the scorer from seeing a torn write.

**Drawback:** during the scatter window (dirty=1), the document is invisible to scorers. For an embedding of dimension 512 this window is short, but it is nonzero. Any score request that arrives while the write is in flight simply omits that document from its results, as if it did not exist.

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

## Strategy Analysis

| | Race condition | Concurrency | Doc visibility | Carry secondary on upsert | Map lookup in score |
|---|---|---|---|---|---|
| Naive | ✅ (global mutex) | ❌ fully serialized | ✅ no (never dirty) | ✅ no | ✅ no |
| Overwrite | ✅ (dirty-bit fence) | ✅ score overlaps write | ❌ dirty during emb+scalar scatter | ✅ no | ✅ no |
| COW Eager | ✅ (new rowIdx + readMutex for scalars) | ✅ score overlaps emb write | ✅ old row visible until flip | ❌ yes (scalars copied to new rowIdx before flip) | ✅ no |
| COW Lazy | ✅ (new rowIdx) | ⚠️ score serializes on scalar sync | ✅ old row visible until flip | ✅ no (scalars always synced from CPU in score) | ❌ yes (rowIdx→docId→scalar per target row) |

**WorkerOverwrite** handles races well and keeps the map stable, but the dirty-bit fence during both primary (emb) and secondary (scalar) updates makes documents transiently invisible to scorers.

**WorkerCopyOnWriteEager** eliminates the visibility problem for primary updates by writing to a fresh rowIdx and flipping atomically. For secondary (scalar) updates, it avoids dirty-bit fencing entirely — the document stays visible — but must hold `m_readMutex` during the scatter to prevent scorers from reading mid-write. The tradeoff is constant map churn: every primary upsert allocates a new rowIdx and frees the old one.

**WorkerCopyOnWriteLazy** avoids the scalar-carry step on the upsert path by keeping scalars on CPU only. But this pushes the cost into every score call: `score` must walk `rowIdx → docId → scalar` for each of the 10k target rows, build a scatter buffer, and H2D-sync it before the score kernel can run. This CPU map lookup on the hot scoring path is the dominant bottleneck — it drives score latency from ~0.3ms to ~8ms and limits throughput to ~126 calls/sec.

## Concurrency Summary

| Worker | Upsert vs Score | Upsert vs Upsert | Scalar update vs Score |
|---|---|---|---|
| Naive | Serialized | Serialized | Serialized |
| Overwrite | Overlapping (dirty bit) | **Races** on m_writeStream | Overlapping (dirty bit) |
| COW Eager | Overlapping (new rowIdx) | Serialized (m_writeMutex) | Serialized (m_readMutex, no dirty-bit fencing) |
| COW Lazy | Overlapping (new rowIdx) | Serialized (m_writeMutex) | Score syncs CPU→GPU per call |

> **Note on Upsert vs Upsert:** The benchmark uses a single upsert thread, so concurrent upserts never occur in practice. WorkerOverwrite would race if two upsert threads overlapped in the GPU scatter phase, since `m_writeMutex` is released before GPU work. COW workers serialize the entire write operation under `m_writeMutex` (including GPU work), so concurrent upserts are safe — but only because of the mutex, not because of rowIdx isolation.
>
> **Note on Upsert vs UpdateScalar in WorkerOverwrite:** Both operations release `m_writeMutex` before GPU work, so the upsert thread and the updateScalar thread (which ARE concurrent in the benchmark) can both issue to `m_writeStream` simultaneously. The dirty-bit kernels are serialized by `m_readMutex`, but the H2D and scatter phases between them are not. If the two operations happen to target the same doc, one thread's `kn_setDirty(CLEAN)` could overwrite the other's `kn_setDirty(DIRTY)` before the scatter completes, briefly exposing a partially-written row. Fixing this would require a dedicated stream mutex or extending `m_writeMutex` to cover all GPU work.

## Benchmark

The benchmark (`test/test_async_memcpy_benchmark.cu`) runs each worker under a mixed workload for 3 seconds using three concurrent threads:

- **Score thread**: issues scoring requests at 3000 calls/sec, each scoring 10k randomly selected rows
- **Upsert+Delete thread**: upserts at 70k docs/sec in batches of 1000; half of each batch are existing docs (re-upserted) and half are new docs that are deleted immediately after to keep the index size stable
- **UpdateScalar thread**: updates scalars at 30k docs/sec in batches of 1000

The index is pre-loaded with 1M documents (out of a 2M capacity) before benchmarking begins.

Reported metrics: mean / p50 / p99 latency per operation, and observed throughput (calls/sec for score, docs/sec for upsert/delete/updateScalar).

## Sample Results

Config: 2M max docs, 1M bootstrapped docs, embDim=512, numScalars=32, updateBatchSize=1000, durationSec=3.
Score: 3000 calls/sec (10k rows each). Upsert: 70k docs/sec. UpdateScalar: 30k docs/sec.

```
=== WorkerNaive ===
  headRowIdx: 1000000 -> 1000500
  score             calls=6574    docs=6574        mean=0.456ms  p50=0.287ms  p99=4.947ms
  upsert            calls=211     docs=211000      mean=13.255ms  p50=13.048ms  p99=15.973ms
  delete            calls=211     docs=105500      mean=0.076ms  p50=0.074ms  p99=0.108ms
  updateScalar      calls=91      docs=91000       mean=2.442ms  p50=1.573ms  p99=6.765ms
  observed QPS (calls):  score=2191.3
  observed doc QPS:  upsert=70333  delete=35167  updateScalar=30333

=== WorkerOverwrite ===
  headRowIdx: 1000000 -> 1000500
  score             calls=8999    docs=8999        mean=0.321ms  p50=0.287ms  p99=1.252ms
  upsert            calls=211     docs=211000      mean=13.404ms  p50=13.266ms  p99=15.715ms
  delete            calls=211     docs=105500      mean=0.290ms  p50=0.264ms  p99=0.469ms
  updateScalar      calls=91      docs=91000       mean=2.492ms  p50=1.754ms  p99=5.566ms
  observed QPS (calls):  score=2999.7
  observed doc QPS:  upsert=70333  delete=35167  updateScalar=30333

=== WorkerCopyOnWriteEager ===
  headRowIdx: 1000000 -> 1000500
  score             calls=8875    docs=8875        mean=0.337ms  p50=0.288ms  p99=1.400ms
  upsert            calls=199     docs=199000      mean=15.142ms  p50=14.615ms  p99=19.711ms
  delete            calls=199     docs=99500       mean=0.301ms  p50=0.269ms  p99=1.168ms
  updateScalar      calls=91      docs=91000       mean=2.496ms  p50=2.035ms  p99=6.511ms
  observed QPS (calls):  score=2958.3
  observed doc QPS:  upsert=66333  delete=33167  updateScalar=30333

=== WorkerCopyOnWriteLazy ===
  headRowIdx: 1000000 -> 1000500
  score             calls=377     docs=377         mean=7.975ms  p50=7.867ms  p99=9.407ms
  upsert            calls=125     docs=125000      mean=24.056ms  p50=23.682ms  p99=31.751ms
  delete            calls=125     docs=62500       mean=8.040ms  p50=7.815ms  p99=15.438ms
  updateScalar      calls=91      docs=91000       mean=1.042ms  p50=0.779ms  p99=4.620ms
  observed QPS (calls):  score=125.7
  observed doc QPS:  upsert=41667  delete=20833  updateScalar=30333
```

**Key observations:**

- **WorkerNaive** hits a score bottleneck: the global mutex means upserts (13ms each) block score threads, limiting score to 2191 calls/sec vs the 3000 target.
- **WorkerOverwrite** reaches full score throughput (3000/sec) by using two separate mutexes — score and upsert can now overlap. Upsert latency is similar to Naive since it still writes in-place with dirty-bit fencing.
- **WorkerCopyOnWriteEager** also reaches near-full score throughput (2958/sec). Upsert is slightly slower than Overwrite (~15ms vs ~13ms) because it must carry scalars to the new rowIdx before flipping the dirty bit.
- **WorkerCopyOnWriteLazy** suffers severely: `score` now syncs CPU scalars to GPU for all 10k target rows on every call, driving score latency to ~8ms and throughput to only 126 calls/sec — 24x slower than Overwrite/COWEager. Upsert is also slower because it must wait while score holds the read mutex during the scalar sync. `updateScalarData` is the fastest of all workers for that operation since it only touches CPU memory.

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
