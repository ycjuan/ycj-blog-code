# Concurrent Read / Write Design Paradigm

Benchmarks five concurrent read/write strategies for a GPU vector index under mixed workloads: simultaneous scoring (reads) and upsert/delete/scalar-update (writes).

## Motivation

A GPU vector index stores embedding vectors in device memory. Concurrent writes (upserts, deletes, scalar updates) and reads (scoring) must be carefully coordinated to avoid serving stale or partially-written data. The naive approach serializes everything through a single mutex — simple but slow. This benchmark explores progressively more concurrent strategies and measures the tradeoff between correctness overhead and throughput.

## The Data Model

### docId and rowIdx

Every document has two identifiers that serve different purposes:

- **`docId`** (`long`) — the stable, caller-facing identity of a document. Assigned by the caller and never changes for the lifetime of the document. Used to upsert, delete, or update a specific document by name.
- **`rowIdx`** (`int`) — the physical slot in the GPU arrays where the document's data currently lives. Assigned internally by the index and can change (in copy-on-write strategies, a re-upsert moves a document to a fresh `rowIdx`). The caller never sees `rowIdx` directly.

The index maintains a CPU-side hash map `docId → rowIdx` to translate between the two. Freed `rowIdx` slots are recycled via a free-list so GPU memory stays compact.

Scoring operates entirely in terms of `rowIdx`: the caller resolves which documents to score (and their current `rowIdx` values) before calling `score`, which then reads embeddings and scalars directly from the GPU arrays by index.

### GPU Arrays

All GPU arrays use **row-major** layout. Each document has:
- An embedding vector (`bfloat16`, dimension 512) stored in a flat GPU array `m_data[rowIdx * embDim ... (rowIdx+1)*embDim]`
- A list of scalar values (`float32`, 32 per doc) stored in `m_d_scalars[rowIdx * numScalars + scalarIdx]`
- A dirty bit `m_d_dirty[rowIdx]` (`DirtyBit::CLEAN` or `DirtyBit::DIRTY`) that marks a row as invisible to scorers

Scoring takes a query embedding and a list of target `rowIdx` values, skips rows where `dirty==1`, and returns dot-product scores weighted by a selected scalar dimension (`targetScalarIdx`).

There are two types of writes:
- **Primary update (upsert):** replaces a document's embedding. This is the expensive operation — 512 bfloat16 values must be transferred to the GPU.
- **Secondary update (scalar):** updates a document's scalar weights without touching the embedding. Much cheaper, but must still be coordinated with concurrent scorers reading the same scalar slots.

## API Design

Write operations (`upsertDocs`, `updateScalarData`, `deleteDocs`) take `docId` as input — the stable caller-facing identity. The index resolves `docId → rowIdx` internally before touching GPU memory.

Score, however, takes `rowIdx` directly. The reason is performance: resolving `docId → rowIdx` requires a CPU hash map lookup per document, which is expensive at scale. A system that scores 10k documents per call and needs sub-millisecond latency cannot afford that overhead on the hot path.

The assumption is that the surrounding system is designed to operate in terms of `rowIdx` when issuing score requests — for example, by maintaining its own `docId → rowIdx` mapping or by receiving `rowIdx` values from a prior retrieval stage. Exactly how to design such a system is out of scope for this post.

## Design Challenges

Getting concurrent reads and writes correct requires navigating four tensions:

1. **Race condition** — A scorer and a writer touching the same GPU memory location from different CUDA streams have no ordering guarantee. Without explicit coordination, the scorer can read a partially-written value.

2. **Concurrency** — If every operation must wait for every other, throughput suffers. The goal is to let scores and writes overlap as much as possible.

3. **Doc visibility** — During an update, a document may need to be temporarily hidden from scorers (dirty=1) to prevent torn reads. Any score request that arrives during this window gets results as if the document does not exist.

4. **docId↔rowIdx map mutation** — Some strategies (copy-on-write) allocate a new rowIdx on every upsert and free the old one. This keeps the embedding write invisible until committed, but it means the map is constantly changing, which adds overhead and complexity.

## The Five Strategies

### WorkerNaive

One global mutex serializes all operations (upserts, deletes, scalar updates, and scoring). Simple and correct but every operation blocks every other.

### WorkerSplitLock

Uses four fine-grained mutexes — one per protected resource — with no dirty bits:
- `m_mapMutex`: protects the CPU `docId → rowIdx` map during resolution
- `m_embDataMutex`: serializes GPU embedding scatter vs. score kernel
- `m_scalarMutex`: serializes GPU scalar scatter vs. score kernel
- `m_dirtyBitMutex`: serializes GPU dirty-bit writes vs. score kernel

H2D transfers happen outside all locks. Each writer holds only the mutex for its own GPU resource, so upsert, updateScalar, and delete can run concurrently with each other. Score holds all three GPU mutexes simultaneously via `std::scoped_lock`. Because there are no dirty bits, score blocks for the full duration of any concurrent scatter kernel on its resource.

### WorkerOverwrite

Uses two mutexes:
- `m_mapMutex`: protects the `docId → rowIdx` map during resolution
- `m_dirtyBitMutex`: protects dirty bits during score reads

Upsert writes in-place to the existing rowIdx. To prevent a scorer from reading a half-written embedding, the dirty bit is set to 1 before the scatter and cleared to 0 after. This means score and upsert can overlap — the scorer sees the row as dirty and skips it — but two concurrent upserts to the same row are still safe because the dirty bit prevents the scorer from seeing a torn write.

**Drawback:** during the scatter window (dirty=1), the document is invisible to scorers. For an embedding of dimension 512 this window is short, but it is nonzero. Any score request that arrives while the write is in flight simply omits that document from its results, as if it did not exist.

### WorkerCopyOnWriteEager

Uses three mutexes:
- `m_mapMutex`: protects the `docId → rowIdx` map and CPU scalar mirror
- `m_dirtyBitMutex`: protects dirty-bit flips vs. score kernel
- `m_scalarMutex`: protects GPU scalar scatter vs. score kernel

Instead of overwriting the existing row, upsert always allocates a **new** rowIdx for the document, scatters the new embedding there, then atomically flips dirty bits: mark the old row dirty (hidden) and mark the new row clean (visible).

This means the scorer never sees a partially-written embedding — either it reads the old clean row or the new clean row, never a half-written one. The flip is the only moment that requires holding `m_dirtyBitMutex`.

Scalar data is kept both on CPU (in `m_docId2scalar`) and on GPU (`m_d_scalars`). `updateScalarData` scatters scalars to GPU immediately under `m_scalarMutex`. `upsertDocs` carries the CPU-side scalars to the new rowIdx between the emb scatter and the dirty-bit flip, so the new row has up-to-date scalar values when it becomes visible.

### WorkerCopyOnWriteLazy

Uses two mutexes:
- `m_mapMutex`: protects the `docId → rowIdx` map and CPU scalar mirror
- `m_dirtyBitMutex`: protects dirty-bit flips vs. score kernel

Same copy-on-write approach for embeddings. The difference is in how scalars are handled:

- `updateScalarData` stores scalars on CPU only — no GPU scatter at all.
- `upsertDocs` does not touch scalars; the new row's scalar slot in `m_d_scalars` is intentionally left stale.
- `score` first snapshots the CPU scalar maps under `m_mapMutex` (resolving `rowIdx → docId → scalar` for each target row), then H2D-syncs the snapshot to GPU and runs the score kernel under `m_dirtyBitMutex`.

This eliminates the scalar-carry step from the hot upsert path, but pushes significant cost into every score call: a CPU map walk under `m_mapMutex` plus an H2D transfer before each score kernel.

## Locking Schemes

The pseudo-code below shows the locking structure of each worker. Operations outside any lock block can run concurrently with other threads.

### WorkerNaive

```
score()        { lock(globalMutex) { scoreKernel } }

upsert()       { lock(globalMutex) { resolveMap; H2D; embScatterKernel } }

updateScalar() { lock(globalMutex) { resolveMap; H2D; scalarScatterKernel } }

delete()       { lock(globalMutex) { resolveMap; H2D; setDirtyKernel } }
```

### WorkerSplitLock

```
score()        { lock(embDataMutex, scalarMutex, dirtyBitMutex) { scoreKernel } }

upsert()       { lock(mapMutex) { resolveMap }
                 H2D
                 lock(embDataMutex) { embScatterKernel } }

updateScalar() { lock(mapMutex) { resolveMap }
                 H2D
                 lock(scalarMutex) { scalarScatterKernel } }

delete()       { lock(mapMutex) { resolveMap }
                 H2D
                 lock(dirtyBitMutex) { setDirtyKernel } }
```

### WorkerOverwrite

```
score()        { lock(dirtyBitMutex) { scoreKernel } }

upsert()       { lock(mapMutex) { resolveMap }
                 H2D
                 lock(dirtyBitMutex) { setDirty(DIRTY) }
                 embScatterKernel
                 lock(dirtyBitMutex) { setDirty(CLEAN) } }

updateScalar() { lock(mapMutex) { resolveMap }
                 H2D
                 lock(dirtyBitMutex) { setDirty(DIRTY) }
                 scalarScatterKernel
                 lock(dirtyBitMutex) { setDirty(CLEAN) } }

delete()       { lock(mapMutex) { resolveMap }
                 H2D
                 lock(dirtyBitMutex) { setDirtyKernel(DIRTY) } }
```

### WorkerCopyOnWriteEager

```
score()        { lock(dirtyBitMutex, scalarMutex) { scoreKernel } }

upsert()       { lock(mapMutex) { resolveMap
                                  H2D
                                  embScatterKernel
                                  carryScalarsH2D
                                  scalarScatterKernel }
                 lock(dirtyBitMutex) { setDirty(old=DIRTY)
                                       setDirty(new=CLEAN) } }

updateScalar() { lock(mapMutex) { resolveMap; updateCPUScalars }
                 H2D
                 lock(scalarMutex) { scalarScatterKernel } }

delete()       { lock(mapMutex) { resolveMap; eraseCPUScalars }
                 H2D
                 lock(dirtyBitMutex) { setDirtyKernel(DIRTY) } }
```

### WorkerCopyOnWriteLazy

```
score()        { lock(mapMutex) { snapshotScalarsFromCPU }
                 lock(dirtyBitMutex) { H2D; scalarScatterKernel; scoreKernel } }

upsert()       { lock(mapMutex) { resolveMap; H2D; embScatterKernel }
                 lock(dirtyBitMutex) { setDirty(old=DIRTY)
                                       setDirty(new=CLEAN) } }

updateScalar() { lock(mapMutex) { updateCPUScalarsOnly } }

delete()       { lock(mapMutex) { resolveMap; eraseCPUScalars }
                 H2D
                 lock(dirtyBitMutex) { setDirtyKernel(DIRTY) } }
```

## Strategy Analysis

| | Race condition | Concurrency | Doc visibility | Carry secondary on upsert | Map lookup in score |
|---|---|---|---|---|---|
| Naive | ✅ (global mutex) | ❌ fully serialized | ✅ no (never dirty) | ✅ no | ✅ no |
| SplitLock | ✅ (per-resource mutex) | ✅ writers overlap each other; score blocks per-resource | ✅ no (never dirty) | ✅ no | ✅ no |
| Overwrite | ✅ (dirty-bit fence) | ✅ score overlaps write | ❌ dirty during emb+scalar scatter | ✅ no | ✅ no |
| COW Eager | ✅ (new rowIdx + m_scalarMutex for scalars) | ✅ score overlaps emb write | ✅ old row visible until flip | ❌ yes (scalars copied to new rowIdx before flip) | ✅ no |
| COW Lazy | ✅ (new rowIdx) | ⚠️ score serializes on scalar sync | ✅ old row visible until flip | ✅ no (scalars always synced from CPU in score) | ❌ yes (rowIdx→docId→scalar per target row) |

**WorkerSplitLock** gives each GPU resource its own mutex, allowing writers to run concurrently with each other. Score still blocks during any scatter kernel touching its resources — there are no dirty bits to allow overlap. It is a strict improvement over Naive for multi-writer workloads.

**WorkerOverwrite** handles races well and keeps the map stable, but the dirty-bit fence during both primary (emb) and secondary (scalar) updates makes documents transiently invisible to scorers.

**WorkerCopyOnWriteEager** eliminates the visibility problem for primary updates by writing to a fresh rowIdx and flipping atomically. For secondary (scalar) updates, it avoids dirty-bit fencing entirely — the document stays visible — but must hold `m_scalarMutex` during the scatter to prevent scorers from reading mid-write. The tradeoff is constant map churn: every primary upsert allocates a new rowIdx and frees the old one.

**WorkerCopyOnWriteLazy** avoids the scalar-carry step on the upsert path by keeping scalars on CPU only. But this pushes the cost into every score call: `score` must hold `m_mapMutex` to walk `rowIdx → docId → scalar` for each of the 10k target rows, build a scatter buffer, H2D-sync it, and only then run the score kernel. This map lookup and transfer on the hot scoring path is the dominant bottleneck — it drives score latency from ~0.3ms to ~10ms and limits throughput to ~103 calls/sec. Upsert and delete are also slower because they contend with score on `m_mapMutex`.

## Concurrency Summary

| Worker | Upsert vs Score | Upsert vs Upsert | Scalar update vs Score | Upsert vs Scalar update |
|---|---|---|---|---|
| Naive | Serialized | Serialized | Serialized | Serialized |
| SplitLock | Overlapping (different GPU mutexes) | Serialized (m_mapMutex) | Overlapping (different GPU mutexes) | ✅ fully overlapping |
| Overwrite | Overlapping (dirty bit) | **Races** on m_writeStream | Overlapping (dirty bit) | ⚠️ races on m_writeStream |
| COW Eager | Overlapping (new rowIdx) | Serialized (m_mapMutex) | Serialized (m_scalarMutex, no dirty-bit fencing) | Serialized (m_mapMutex) |
| COW Lazy | Overlapping (new rowIdx) | Serialized (m_mapMutex) | Score syncs CPU→GPU per call | ✅ fully overlapping |

> **Note on Upsert vs Upsert:** The benchmark uses a single upsert thread, so concurrent upserts never occur in practice. WorkerOverwrite would race if two upsert threads overlapped in the GPU scatter phase, since `m_mapMutex` is released before GPU work. COW workers serialize the entire write operation under `m_mapMutex` (including GPU work), so concurrent upserts are safe — but only because of the mutex, not because of rowIdx isolation.
>
> **Note on Upsert vs UpdateScalar in WorkerOverwrite:** Both operations release `m_mapMutex` before GPU work, so the upsert thread and the updateScalar thread (which ARE concurrent in the benchmark) can both issue to `m_writeStream` simultaneously. The dirty-bit kernels are serialized by `m_dirtyBitMutex`, but the H2D and scatter phases between them are not. If the two operations happen to target the same doc, one thread's `kn_setDirty(CLEAN)` could overwrite the other's `kn_setDirty(DIRTY)` before the scatter completes, briefly exposing a partially-written row. Fixing this would require a dedicated stream mutex or extending `m_mapMutex` to cover all GPU work.

## Benchmark

The benchmark (`test/test_concurrent_read_write_paradigm.cu`) runs each worker under a mixed workload for 15 seconds using three concurrent threads:

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
  score             calls=13492                     mean=1.111ms       p50=0.972ms       p99=5.420ms
  upsert            calls=285     docs=285000      mean=52.699ms      p50=50.327ms      p99=60.298ms
  delete            calls=285     docs=142500      mean=0.082ms       p50=0.072ms       p99=0.129ms
  updateScalar      calls=451     docs=451000      mean=4.416ms       p50=4.344ms       p99=8.830ms
  observed QPS (calls):  score=899.5
  observed doc QPS:  upsert=19000.0  delete=9500.0  updateScalar=30066.7

=== WorkerSplitLock ===
  headRowIdx: 1000000 -> 1000500
  score             calls=15843                     mean=0.946ms       p50=0.899ms       p99=1.797ms
  upsert            calls=294     docs=294000      mean=51.157ms      p50=49.752ms      p99=61.319ms
  delete            calls=294     docs=147000      mean=0.124ms       p50=0.071ms       p99=0.692ms
  updateScalar      calls=451     docs=451000      mean=4.560ms       p50=4.438ms       p99=7.953ms
  observed QPS (calls):  score=1056.2
  observed doc QPS:  upsert=19600.0  delete=9800.0  updateScalar=30066.7

=== WorkerOverwrite ===
  headRowIdx: 1000000 -> 1000500
  score             calls=16198                     mean=0.926ms       p50=0.897ms       p99=1.755ms
  upsert            calls=299     docs=299000      mean=50.288ms      p50=49.835ms      p99=60.621ms
  delete            calls=299     docs=149500      mean=0.116ms       p50=0.066ms       p99=0.686ms
  updateScalar      calls=451     docs=451000      mean=4.823ms       p50=4.632ms       p99=8.066ms
  observed QPS (calls):  score=1079.9
  observed doc QPS:  upsert=19933.3  delete=9966.7  updateScalar=30066.7

=== WorkerCopyOnWriteEager ===
  headRowIdx: 1000000 -> 1000500
  score             calls=15170                     mean=0.988ms       p50=0.933ms       p99=1.885ms
  upsert            calls=276     docs=276000      mean=54.382ms      p50=51.743ms      p99=64.133ms
  delete            calls=276     docs=138000      mean=0.373ms       p50=0.107ms       p99=2.098ms
  updateScalar      calls=451     docs=451000      mean=4.986ms       p50=4.866ms       p99=9.912ms
  observed QPS (calls):  score=1011.3
  observed doc QPS:  upsert=18400.0  delete=9200.0  updateScalar=30066.7

=== WorkerCopyOnWriteLazy ===
  headRowIdx: 1000000 -> 1000500
  score             calls=1552                      mean=9.666ms       p50=9.140ms       p99=14.321ms
  upsert            calls=241     docs=241000      mean=62.427ms      p50=61.799ms      p99=73.911ms
  delete            calls=241     docs=120500      mean=8.550ms       p50=8.513ms       p99=11.318ms
  updateScalar      calls=451     docs=451000      mean=7.302ms       p50=6.973ms       p99=14.604ms
  observed QPS (calls):  score=103.5
  observed doc QPS:  upsert=16066.7  delete=8033.3  updateScalar=30066.7
```

**Key observations:**

- **WorkerNaive** hits a score bottleneck: the global mutex means upserts (~53ms each) block score threads, limiting score to 900 calls/sec vs the 3000 target.
- **WorkerSplitLock** improves score throughput to 1056/sec — better than Naive since score no longer blocks during map resolution or H2D. However, score still blocks during scatter kernels (no dirty bits), so throughput is similar to WorkerOverwrite rather than dramatically better.
- **WorkerOverwrite** reaches 1080/sec, slightly ahead of SplitLock. Dirty bits allow the score kernel and emb scatter to overlap on different rows, whereas SplitLock blocks score for the entire scatter duration.
- **WorkerCopyOnWriteEager** scores at 1011/sec, slightly behind Overwrite. Upsert is slower (~54ms vs ~50ms) because it must carry scalars to the new rowIdx before flipping the dirty bit.
- **WorkerCopyOnWriteLazy** suffers severely: `score` holds `m_mapMutex` to walk `rowIdx → docId → scalar` for all 10k target rows, H2D-syncs the result, then scores — driving score latency to ~10ms and throughput to only 104 calls/sec, ~10x slower than the others. Upsert and delete are slower too because they contend with score on `m_mapMutex`.

## Build

```bash
mkdir -p concurrent_read_write_paradigm/build
cmake -S concurrent_read_write_paradigm -B concurrent_read_write_paradigm/build
cmake --build concurrent_read_write_paradigm/build
```

## Run

```bash
./concurrent_read_write_paradigm/run.sh
```

or directly:

```bash
./concurrent_read_write_paradigm/build/test_concurrent_read_write_paradigm
```
