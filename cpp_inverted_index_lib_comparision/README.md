# C++ inverted index library comparison

Compares three C++ approaches for one specific scenario: a query that is a
fixed-size tuple of terms (e.g. `"vocA vocB vocC"`), looked up as a single
atomic *key* rather than as three independent terms to AND together. All
code lives in this directory; the "Benchmark" section below has full
methodology and example results.

| | Hand-rolled ("PISA-style") | Xapian | KeyEngine |
|---|---|---|---|
| Library | hand-rolled, modeled after [PISA](https://github.com/pisa-engine/pisa)'s core concepts | real [Xapian](https://xapian.org) | hand-rolled, no inverted index at all |
| Term dictionary / posting lists | custom `TermDictionary` + `vector<vector<int>>` | owned internally by Xapian | n/a -- plain `unordered_map<string, vector<docId>>` |
| How the key is answered | classic boolean AND of the key's terms (`vocA AND vocB AND vocC`), via custom merge functions | same AND query, via `Xapian::Query::OP_AND` | exact multi-term key lookup -- no tokenization, no posting-list merge |
| Ranking | none (not attempted) | explicitly disabled via `Xapian::BoolWeight` (Xapian supports ranking/BM25 natively, we just don't use it here) | none -- not a ranking structure |
| Dependencies | none (single `g++` call) | requires building/installing libxapian | none |

## Why these three?

PISA-style and Xapian represent the "classic inverted index" answer to this
scenario: decompose the key into its terms and intersect their posting
lists with a boolean AND. `pisa_engine.h` is a minimal from-scratch
posting-list index with a merge-based AND; `xapian_engine.h` builds the
equivalent `Xapian::Query` (`OP_AND`) against a real, mature IR library, to
show the same approach backed by production-grade code.

`KeyEngine` targets the narrower use case directly: when the whole
multi-term tuple is itself the atomic lookup key, there's no need for
tokenization or posting lists at all -- just a single hash-map lookup. This
is **not** semantically equivalent to the AND query over the same terms
(see "Phrase-as-key benchmark: design" below for why), but is dramatically
faster when that exact-key semantics is really what's needed.

## Quick summary

- **Have a fixed-size multi-term key that's always looked up as a whole,
  not via arbitrary AND combinations?** See `key_engine.h` -- a plain hash
  map outperforms both inverted-index approaches by orders of magnitude for
  this case.
- **Want to learn the fundamentals, or need a tiny dependency-free
  component for the AND-query fallback?** See `pisa_engine.h` -- dictionary
  + posting lists + merge-based AND, from scratch.
- **Want the AND-query fallback backed by a production library instead?**
  See `xapian_engine.h` -- a mature, actively maintained C++ library
  (ranking available too, if later needed).

## Benchmark

The benchmark (`main.cpp` + the engine headers) generates a
size-configurable synthetic corpus, builds all three engines over it, then
for a handful of phrases looks each one up via `KeyEngine` and via the
equivalent `vocA AND vocB AND vocC` query on PISA/Xapian, reporting result
counts and p50/p95 latency side by side.

```bash
PATH=/usr/local/bin:$PATH ./run.sh --num_docs=200000 --phrase_card=5000
```

### Corpus parameters

| Flag | Short | Meaning | Default |
|---|---|---|---|
| `--num_docs` | `-d` | number of documents | 20000 |
| `--num_char_per_vocab` | `-l` | character length of each distinct vocabulary term | 6 |
| `--phrase_card` | `-k` | phrase cardinality (# of distinct seed phrases) | 2000 |
| `--reps` | `-r` | repetitions per query; reports p50/p95 latency across them | 20 |
| `--num_vocab_per_phrase` | `-p` | terms per seed phrase | 3 |
| `--num_phrases_per_doc` | `-n` | seed phrases sampled into each doc | 5 |

Long form takes `--flag=value`; short form takes `-f value` (e.g.
`-d 200000 -k 5000`). Run `./benchmark --help` for the full usage message.

Generation is seeded (`seed=42`) for reproducibility. A fixed pool of
`phrase_card` distinct *seed phrases* is generated up front (each a
`num_vocab_per_phrase`-term tuple, each term `num_char_per_vocab`
characters). Every doc is then built by sampling `num_phrases_per_doc` of
those *same* seed phrases (with replacement) -- so a given phrase recurs
across many docs whenever `num_docs` is large relative to `phrase_card`.
PISA/Xapian see each doc as the flat bag of all terms across its sampled
phrases; `KeyEngine` sees each doc as its list of sampled phrase strings.
Reusing the same seed-phrase pool across docs is what makes the benchmark
meaningful -- a phrase has to actually recur to have anything interesting
to look up (an earlier version of this benchmark sampled each phrase's
terms independently per doc, so phrases almost never repeated and `Key#`
was ~1 for nearly every query).

Keys for the benchmark queries are the first 5 distinct phrases encountered
in the generated corpus (guaranteed to exist).

### Build & run

```bash
PATH=/usr/local/bin:$PATH ./run.sh
# or with custom corpus parameters (long or short flags):
PATH=/usr/local/bin:$PATH ./run.sh --num_docs=200000 --phrase_card=5000 --num_char_per_vocab=8 --reps=20
PATH=/usr/local/bin:$PATH ./run.sh -d 200000 -k 5000 -l 8 -r 20
```

Each row reports p50 and p95 latency (milliseconds, across `--reps`
repetitions) and the implied QPS from p50 (`1e6 / p50_latency_us`) -- i.e.
single-threaded queries/sec implied by one query's latency, not a
multi-threaded throughput measurement. A higher `--reps` gives a more stable
p95 estimate (the default of 20 is a reasonable floor; the included 20k/200k
examples below use reps=20).

The `*Copy` column reports p50 latency for copying the query's result
doc-id vector out of the engine into a fresh `vector<int>` (measured
separately from the lookup/merge compute time above). All engines here are
CPU-only, so this is a same-host copy -- but it isolates the "transfer the
result out" cost from the actual query compute. It scales with result size,
not query complexity.

If a CUDA toolkit (`nvcc`) is available at build time, the Makefile also
compiles [`cuda_copy.cu`](cuda_copy.cu) and the binary always additionally
measures a real GPU D2H copy per query (no runtime flag needed): each
query's result vector is uploaded once (untimed) into real device memory
via `cudaMalloc`+`cudaMemcpy` (host-to-device), and a `*GPUCopy` column
reports the p50 latency of a real `cudaMemcpy(..., cudaMemcpyDeviceToHost)`
call copying it back out -- an actual device-to-host transfer, not a
same-host analogy. This requires a CUDA-capable GPU at runtime
(`nvidia-smi` should show one). Without `nvcc` at build time, the binary
still builds and runs fine; it just omits the `*GPUCopy` columns. Because
these result sets are small (tens to hundreds of thousands of ints), the
measured GPU copy time is dominated by fixed `cudaMemcpy` launch/sync
overhead (~0.003-0.008 ms here) rather than PCIe bandwidth -- it only starts
to track transfer size once results reach the hundreds-of-thousands range.

### Example results

20,000 docs, phrase_card=2000, 3 terms/phrase, 5 phrases/doc (default
settings), on a Tesla T4 (real `cudaMemcpy` D2H GPU copy, always measured
when built with CUDA):

```
Index build time: pisa=40.49 ms, xapian=404.14 ms, key_engine=8.37 ms

Key                          Key p50   Key p95   Key QPS  Key Copy Key GPUCopy  PISA p50  PISA p95  PISA QPS PISA Copy PISA GPUCopy Xapian p50 Xapian p95 Xapian QPS XapianCopy Xapian GPUCopy   Key#   PISA#  Xapian#
                                (ms)      (ms)             p50(ms)     p50(ms)      (ms)      (ms)             p50(ms)      p50(ms)       (ms)       (ms)               p50(ms)        p50(ms)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
dhovhb bubpdo blthmm          0.0001    0.0001  14285714    0.0000      0.0068    0.0008    0.0010   1240695    0.0000       0.0067     0.0100     0.0229     100050     0.0000         0.0070     53      53       53
lekece rrslsz sfaiyz          0.0001    0.0002  14285714    0.0000      0.0068    0.0008    0.0013   1280410    0.0000       0.0068     0.0101     0.0128      98824     0.0000         0.0071     52      52       52
pswert lzvuxa dpnshv          0.0001    0.0002  12820513    0.0000      0.0070    0.0008    0.0010   1240695    0.0000       0.0070     0.0106     0.0121      94038     0.0000         0.0083     55      55       55
aavfss suupby jqdxww          0.0001    0.0002   9708738    0.0000      0.0085    0.0012    0.0016    805802    0.0000       0.0082     0.0202     0.0211      49397     0.0000         0.0081     58      58       58
nagyec vudxmy ywehzi          0.0001    0.0002   8620690    0.0001      0.0094    0.0016    0.0022    629723    0.0001       0.0096     0.0169     0.0193      59067     0.0001         0.0085     42      42       42
```

200,000 docs, phrase_card=5000, 8-char terms:

```bash
PATH=/usr/local/bin:$PATH ./run.sh -d 200000 -k 5000 -l 8
```

```
Index build time: pisa=481.76 ms, xapian=4666.68 ms, key_engine=113.92 ms

Key                          Key p50   Key p95   Key QPS  Key Copy Key GPUCopy  PISA p50  PISA p95  PISA QPS PISA Copy PISA GPUCopy Xapian p50 Xapian p95 Xapian QPS XapianCopy Xapian GPUCopy   Key#   PISA#  Xapian#
                                (ms)      (ms)             p50(ms)     p50(ms)      (ms)      (ms)             p50(ms)      p50(ms)       (ms)       (ms)               p50(ms)        p50(ms)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
awdcutwm zxdjnsct efe...      0.0001    0.0005  11904762    0.0000      0.0065    0.0017    0.0022    603865    0.0000       0.0065     0.0287     0.0448      34844     0.0000         0.0064    191     191      191
pspxuzvh aytmuhsz swb...      0.0001    0.0002  12048193    0.0000      0.0063    0.0015    0.0029    672495    0.0000       0.0063     0.0256     0.0277      39096     0.0000         0.0067    162     162      162
vgmgruty ppjedbqg tee...      0.0001    0.0001  11764706    0.0000      0.0065    0.0018    0.0022    549753    0.0000       0.0065     0.0321     0.0434      31170     0.0000         0.0063    216     216      216
vyehomkt oyqphjxz yac...      0.0001    0.0003  12987013    0.0001      0.0084    0.0017    0.0019    604595    0.0001       0.0077     0.0298     0.0409      33574     0.0000         0.0081    194     194      194
qrlsvlxl petjbzoh ttm...      0.0002    0.0004   6493506    0.0001      0.0080    0.0028    0.0035    363372    0.0001       0.0080     0.0600     0.0710      16674     0.0001         0.0081    216     216      216
```

Note that `Key#`/`PISA#`/`Xapian#` all grow with `num_docs / phrase_card`
(e.g. ~42-58 at 20k docs, close to the expected `20000 * 5 / 2000 = 50`) --
each seed phrase recurs across roughly that many docs, which is the whole
point of sampling from a fixed phrase pool instead of generating
independent random terms per doc. The `*GPUCopy` columns' higher, more
uniform floor (fixed launch/sync overhead) compared to the same-host
`*Copy` columns is visible even at these small result sizes.

### Phrase-as-key benchmark: design

This benchmark compares three ways of answering "does this exact key
exist, and if so in which docs":

- [`key_engine.h`](key_engine.h) — `KeyEngine`: a plain
  `unordered_map<string, vector<docId>>` keyed by the *entire* key string.
  No tokenization, no posting-list merge -- a single O(1) average-case hash
  lookup per query.
- `PisaEngine`/`XapianEngine` answering the equivalent classic boolean
  query `vocA AND vocB AND vocC`, i.e. decomposing the key into its terms
  and intersecting their posting lists.

These are **not** semantically equivalent, by design: `KeyEngine` only
matches docs that contain that exact key (the terms together, as
generated), whereas the AND query matches any doc containing all three
terms anywhere -- even if they come from different keys in that doc. So in
general `Key#` is expected to be `<= PISA#`/`Xapian#`. In this benchmark's
corpus, every seed phrase's terms are generated fresh and distinct from
every other phrase's terms (see "Corpus parameters" above), so that gap
never actually opens up here -- `Key#` always equals `PISA#`/`Xapian#` in
the example output. The gap would reappear with a corpus where individual
terms (not whole phrases) are reused across different phrases, making it
possible for a doc to incidentally contain all of a key's terms without
containing that exact phrase.

### Takeaways (from the runs above, not a general claim)

- **Phrase-as-key wins by orders of magnitude when it's the right model**:
  `KeyEngine`'s single hash lookup runs in ~0.0001 ms regardless of corpus
  size/cardinality, versus PISA/Xapian needing to intersect real posting
  lists for the equivalent AND query -- a gap that grows as those posting
  lists get longer (e.g. at 200k docs, PISA's AND query took ~0.001-0.0015
  ms and Xapian's took ~0.025-0.03 ms, both far slower than KeyEngine's flat
  ~0.0001 ms). The tradeoff is semantic, not just a performance knob:
  `KeyEngine` can only answer "does this exact key exist", not "find all
  docs containing these terms in any combination" -- use it only when the
  former is really the question being asked.
- **Fixing a phrase-reuse bug**: an earlier version of this benchmark
  generated each phrase's terms independently per doc from a shared term
  vocabulary, so the same exact phrase almost never recurred across docs
  (`Key#` was ~1 for nearly every query, regardless of corpus size). Sampling
  `num_phrases_per_doc` phrases per doc from a *fixed* pool of `phrase_card`
  seed phrases fixes this: `Key#` now scales with `num_docs / phrase_card`
  as expected (see the example above).
- **Copy-out cost is small but not zero, and scales with result size, not
  query complexity**: in these runs, copying a handful of matching doc ids
  is unmeasurably fast (`0.0000 ms`) for the same-host `*Copy` columns. The
  real `cudaMemcpy` D2H transfer (`*GPUCopy` columns, measured whenever
  built with CUDA) has a non-trivial fixed floor (~0.0065-0.008 ms here)
  from kernel launch/sync overhead that dominates at these small result
  sizes -- it only starts to track transfer size (PCIe bandwidth) once
  results reach the hundreds-of-thousands range, a cost worth tracking
  separately from lookup/merge compute time, especially for any design that
  would put the index in device memory.
- **Index build**: PISA-style in-memory `vector<vector<int>>` postings build
  roughly 10x faster than Xapian's `WritableDatabase::add_document`, which
  does more per-term bookkeeping (positions, term-frequency stats,
  on-disk-ready structures) even in `DB_BACKEND_INMEMORY` mode. `KeyEngine`'s
  build (one hash-map insert per doc) is faster still.
- **Both PISA and Xapian agree** on every result set (`PISA#` == `Xapian#`)
  across all tested configurations, confirming the two AND-query
  implementations are semantically equivalent.
- Take absolute numbers with a grain of salt: this is an in-memory,
  single-query microbenchmark on a synthetic corpus, not a production
  workload. Re-run `./run.sh` with parameters matching your real data shape
  before drawing conclusions for a specific use case.
