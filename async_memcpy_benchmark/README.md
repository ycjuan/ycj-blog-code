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

Config: 2M max docs, 1M bootstrapped docs, embDim=512, numScalars=32, updateBatchSize=1000, durationSec=15.
Score: 3000 calls/sec (10k rows each). Upsert: 70k docs/sec. UpdateScalar: 30k docs/sec.

```
=== WorkerNaive ===
  headRowIdx: 1000000 -> 1000500
  score             calls=31141                     mean=0.482ms       p50=0.288ms       p99=5.047ms
  upsert            calls=1044    docs=1044000     mean=14.032ms      p50=13.433ms      p99=16.136ms
  delete            calls=1044    docs=522000      mean=0.077ms       p50=0.074ms       p99=0.115ms
  updateScalar      calls=451     docs=451000      mean=2.347ms       p50=1.498ms       p99=6.150ms
  observed QPS (calls):  score=2076.1
  observed doc QPS:  upsert=69600.0  delete=34800.0  updateScalar=30066.7

=== WorkerOverwrite ===
  headRowIdx: 1000000 -> 1000500
  score             calls=44791                     mean=0.333ms       p50=0.288ms       p99=1.386ms
  upsert            calls=1047    docs=1047000     mean=14.156ms      p50=13.691ms      p99=16.849ms
  delete            calls=1047    docs=523500      mean=0.305ms       p50=0.268ms       p99=0.621ms
  updateScalar      calls=451     docs=451000      mean=2.546ms       p50=1.844ms       p99=5.884ms
  observed QPS (calls):  score=2986.1
  observed doc QPS:  upsert=69800.0  delete=34900.0  updateScalar=30066.7

=== WorkerCopyOnWriteEager ===
  headRowIdx: 1000000 -> 1000500
  score             calls=43735                     mean=0.342ms       p50=0.291ms       p99=1.436ms
  upsert            calls=957     docs=957000      mean=15.679ms      p50=15.222ms      p99=19.359ms
  delete            calls=957     docs=478500      mean=0.537ms       p50=0.272ms       p99=1.888ms
  updateScalar      calls=451     docs=451000      mean=3.203ms       p50=2.141ms       p99=7.863ms
  observed QPS (calls):  score=2915.7
  observed doc QPS:  upsert=63800.0  delete=31900.0  updateScalar=30066.7

=== WorkerCopyOnWriteLazy ===
  headRowIdx: 1000000 -> 1000500
  score             calls=1539                      mean=9.747ms       p50=8.820ms       p99=14.401ms
  upsert            calls=556     docs=556000      mean=26.982ms      p50=27.775ms      p99=31.052ms
  delete            calls=556     docs=278000      mean=8.369ms       p50=8.278ms       p99=10.791ms
  updateScalar      calls=451     docs=451000      mean=6.086ms       p50=6.232ms       p99=13.045ms
  observed QPS (calls):  score=102.6
  observed doc QPS:  upsert=37066.7  delete=18533.3  updateScalar=30066.7
```

**Key observations:**

- **WorkerNaive** hits a score bottleneck: the global mutex means upserts (14ms each) block score threads, limiting score to 2076 calls/sec vs the 3000 target.
- **WorkerOverwrite** reaches near-full score throughput (2986/sec) by using two separate mutexes — score and upsert can now overlap. Upsert latency is similar to Naive since it still writes in-place with dirty-bit fencing.
- **WorkerCopyOnWriteEager** also reaches near-full score throughput (2916/sec). Upsert is slightly slower than Overwrite (~16ms vs ~14ms) because it must carry scalars to the new rowIdx before flipping the dirty bit. UpdateScalar is also slightly slower (~3.2ms vs ~2.5ms) due to holding `m_readMutex` during the scatter to prevent scorers from reading mid-write.
- **WorkerCopyOnWriteLazy** suffers severely: `score` must snapshot CPU scalars for all 10k target rows under `m_writeMutex`, then H2D-sync them before the score kernel, driving score latency to ~10ms and throughput to only 103 calls/sec — 29x slower than Overwrite/COWEager. Upsert and delete are also slower because they contend with score on `m_writeMutex`.

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
