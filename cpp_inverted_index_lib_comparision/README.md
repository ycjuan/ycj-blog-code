# C++ inverted index library comparison

Compares how different C++ approaches handle an inverted index and boolean
(AND/OR/NOT) retrieval over the same synthetic corpus and identical queries.
All code lives in this directory; the "Benchmark" section below has full
methodology and example results.

| | Hand-rolled ("PISA-style") | Xapian | KeyEngine |
|---|---|---|---|
| Library | hand-rolled, modeled after [PISA](https://github.com/pisa-engine/pisa)'s core concepts | real [Xapian](https://xapian.org) | hand-rolled, no inverted index at all |
| Term dictionary / posting lists | custom `TermDictionary` + `vector<vector<int>>` | owned internally by Xapian | n/a -- plain `unordered_map<string, vector<docId>>` |
| Flat AND / OR | custom merge functions | `Xapian::Query::OP_AND` / `OP_OR` | n/a |
| Nested boolean (`(A AND B) OR (NOT C)`) | shared recursive-descent AST (`query_ast.h`), evaluated via merge-based set ops | same shared AST, converted to native `Xapian::Query` trees | n/a -- exact multi-term key lookup only, not boolean |
| Ranking | none (not attempted) | explicitly disabled via `Xapian::BoolWeight` (Xapian supports ranking/BM25 natively, we just don't use it here) | none -- not a ranking structure |
| Dependencies | none (single `g++` call) | requires building/installing libxapian | none |

## Why these three?

PISA is primarily designed for large-scale, SIMD-compressed, *ranked* top-k
retrieval (WAND/MaxScore/BlockMax-WAND over disjunctive term lists) -- it
does not ship a general nested-boolean-expression query layer. To support
queries like `(quick AND dog) OR (NOT fox)` on top of PISA's primitives, you
have to write your own recursive evaluator, so `pisa_engine.h`
demonstrates exactly that: a minimal from-scratch inverted index plus a
shared boolean query parser/evaluator, to show what PISA (or any engine
offering only flat AND/OR primitives) would need glued on top.

Xapian, by contrast, is a general-purpose IR library whose `Xapian::Query`
is natively a *tree* of operators (`OP_AND`, `OP_OR`, `OP_AND_NOT`, `OP_XOR`,
...), so nested boolean expressions are first-class -- no custom parser
required. `xapian_engine.h` converts the identical shared AST directly into
`Xapian::Query` trees for direct comparison.

`KeyEngine` targets a different, narrower use case: when a whole multi-term
tuple (e.g. `"vocA vocB vocC"`) is itself the atomic lookup key, rather than
three terms to be ANDed together. There's no tokenization or posting-list
merge -- just a single hash-map lookup. This is **not** semantically
equivalent to the AND query over the same terms (see the "Phrase-as-key
benchmark: design" section below for why), but is dramatically
faster when that exact-key semantics is really what's needed.

The PISA-style and Xapian engines produce **identical results** for all
boolean queries, confirming the two approaches are semantically equivalent --
the difference is in how much boolean-query machinery you have to write
yourself, and in each library's internal performance characteristics.

## Quick summary

- **Want to learn the fundamentals, or need a tiny dependency-free
  component?** See `pisa_engine.h` -- dictionary + posting lists +
  merge-based set ops from scratch, with a nested-boolean parser
  (`query_ast.h`) layered on top of flat AND/OR primitives.
- **Want pure boolean retrieval in production without writing your own query
  tree evaluator?** See `xapian_engine.h` -- a mature, actively
  maintained C++ library with native nested boolean queries (and ranking
  available if later needed, unlike PISA).
- **Have a fixed-size multi-term key that's always looked up as a whole,
  not via arbitrary AND/OR combinations?** See `key_engine.h` -- a
  plain hash map outperforms both inverted-index approaches by orders of
  magnitude for this narrower case.
- **Want large-scale ranked top-k retrieval over huge compressed indexes?**
  None of these cover this directly, but PISA (full library, not our toy
  re-implementation) is purpose-built for exactly that; Xapian also supports
  ranked search (BM25 and others) if boolean-only isn't the end goal.

## Benchmark

The benchmark (`main.cpp` + the engine headers) generates a
size-configurable synthetic corpus, builds all engines over it, and runs an
identical set of boolean queries (parsed once into a shared AST, then
evaluated by each engine) plus a separate phrase-as-key benchmark, reporting
result counts and p50/p95 latency side by side.

```bash
PATH=/usr/local/bin:$PATH ./run.sh --num_docs=200000 --num_vocab_per_doc=30 --vocab_card=5000
```

### Corpus parameters

| Flag | Short | Meaning | Default |
|---|---|---|---|
| `--num_docs` | `-d` | number of documents | 20000 |
| `--num_vocab_per_doc` | `-v` | terms sampled (with replacement) into each doc | 20 |
| `--num_char_per_vocab` | `-l` | character length of each distinct vocabulary term | 6 |
| `--vocab_card` | `-k` | vocabulary cardinality (# of distinct terms overall) | 2000 |
| `--reps` | `-r` | repetitions per query; reports p50/p95 latency across them | 20 |
| `--num_vocab_per_key` | `-p` | terms per key, for the phrase-as-key benchmark | 3 |
| `--num_keys_per_doc` | `-n` | keys per doc, for the phrase-as-key benchmark | 5 |
| `--gpu_copy` | `-g` | also measure a real `cudaMalloc`+`cudaMemcpy` D2H copy per query | off |

Long form takes `--flag=value`; short form takes `-f value` (e.g.
`-d 200000 -v 30`). Run `./benchmark --help` for the full usage message.

Generation is seeded (`seed=42`) for reproducibility. Each doc is a
space-joined string of `num_vocab_per_doc` terms drawn uniformly from a fixed
vocabulary of `vocab_card` distinct, `num_char_per_vocab`-character terms.
The phrase-as-key benchmark uses a separate corpus (seeded independently,
`seed=43`) that reuses `num_docs`/`num_char_per_vocab`/`vocab_card` but
generates docs as `num_keys_per_doc` multi-term keys instead (see below).

### Design: one query AST, two evaluators

[`query_ast.h`](query_ast.h) parses a query string (grammar: `NOT` > `AND` >
`OR`, with parentheses) into a
`QueryNode` tree exactly once per query. Both engines then evaluate the
*same* tree:

- [`pisa_engine.h`](pisa_engine.h) — `PisaEngine::evaluate(node)` recurses
  over the AST, combining posting lists with merge-based
  intersect/union/difference.
- [`xapian_engine.h`](xapian_engine.h) — `XapianEngine::evaluate(node)`
  recurses over the same AST, building an `Xapian::Query` tree
  (`OP_AND`/`OP_OR`/`OP_AND_NOT`) and executing it with `Xapian::BoolWeight`
  (pure boolean matching, no ranking).

This guarantees both engines are asked literally the same question, so any
difference is purely in each library's internals, not in query construction.
The benchmark also asserts both engines return the identical sorted doc-id
set per query as a correctness check (`Match` column).

### Query set

1. Single term: `v0`
2. Flat AND: `v0 AND v1`
3. AND then OR: `(v0 AND v1) OR v2`
4. Two AND groups OR'd: `(v0 AND v1) OR (v2 AND v3)`
5. AND + OR + NOT mixed: `v0 AND (v1 OR v2) AND NOT v3`
6. Wide flat OR (5 terms): `v0 OR v1 OR v2 OR v3 OR v4`
7. OR group AND NOT an AND group: `(v0 OR v1) AND NOT (v2 AND v3)`

`v0..v4` are the first 5 terms of the generated vocabulary (guaranteed to
exist in the corpus, so results are non-trivial).

### Build & run

```bash
PATH=/usr/local/bin:$PATH ./run.sh
# or with custom corpus parameters (long or short flags):
PATH=/usr/local/bin:$PATH ./run.sh --num_docs=200000 --num_vocab_per_doc=30 --vocab_card=5000 --num_char_per_vocab=8 --reps=20
PATH=/usr/local/bin:$PATH ./run.sh -d 200000 -v 30 -k 5000 -l 8 -r 20
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
not query complexity (e.g. compare the wide-OR query's larger copy time
against the narrow AND query's near-zero one below).

If a CUDA toolkit (`nvcc`) is available at build time, the Makefile also
compiles [`cuda_copy.cu`](cuda_copy.cu) and the binary gains a `--gpu_copy`/
`-g` flag. When passed, each query's result vector is additionally uploaded
once (untimed) into real device memory via `cudaMalloc`+`cudaMemcpy`
(host-to-device), and a `*GPUCopy` column reports the p50 latency of a real
`cudaMemcpy(..., cudaMemcpyDeviceToHost)` call copying it back out -- an
actual device-to-host transfer, not a same-host analogy. This requires a
CUDA-capable GPU at runtime (`nvidia-smi` should show one); without
`--gpu_copy`, the binary behaves exactly as before and needs no GPU. Because
these result sets are small (tens to hundreds of thousands of ints), the
measured GPU copy time is dominated by fixed `cudaMemcpy` launch/sync
overhead (~0.003-0.008 ms here) rather than PCIe bandwidth -- it only starts
to track transfer size once results reach the hundreds-of-thousands range.

### Example results

20,000 docs, 20 terms/doc, 2,000-term vocabulary:

```
Index build time: pisa=46.45 ms, xapian=485.22 ms

Query                                   PISA p50  PISA p95   PISA QPS  PISA Copy  Xapian p50  Xapian p95 Xapian QPS  Xapian Copy    PISA#  Xapian#  Match
                                            (ms)      (ms)               p50(ms)        (ms)        (ms)                 p50(ms)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
juyetu                                    0.0001    0.0005   11111111     0.0000      0.0175      0.0318      57241       0.0000      191      191    yes
juyetu AND ppelec                         0.0007    0.0020    1451379     0.0000      0.0053      0.0085     187758       0.0000        4        4    yes
(juyetu AND ppelec) OR blwipd             0.0016    0.0024     629723     0.0000      0.0254      0.0282      39358       0.0000      198      198    yes
(juyetu AND ppelec) OR (blwipd AN...      0.0015    0.0027     671592     0.0000      0.0120      0.0142      83654       0.0000        6        6    yes
juyetu AND (ppelec OR blwipd) AND...      0.0714    0.0803      14004     0.0000      0.0108      0.0151      92885       0.0000        5        5    yes
juyetu OR ppelec OR blwipd OR sqa...      0.0094    0.0154     106078     0.0001      0.1243      0.1401       8045       0.0001      944      944    yes
(juyetu OR ppelec) AND NOT (blwip...      0.0720    0.0816      13886     0.0001      0.0555      0.0607      18010       0.0001      359      359    yes
```

200,000 docs, 30 terms/doc, 5,000-term vocabulary:

```
Index build time: pisa=695.33 ms, xapian=8415.11 ms

Query                                   PISA p50  PISA p95   PISA QPS  PISA Copy  Xapian p50  Xapian p95 Xapian QPS  Xapian Copy    PISA#  Xapian#  Match
                                            (ms)      (ms)               p50(ms)        (ms)        (ms)                 p50(ms)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
juyetupp                                  0.0002    0.0008    6250000     0.0001      0.1175      0.1608       8510       0.0001     1194     1194    yes
juyetupp AND elecblwi                     0.0044    0.0120     226501     0.0000      0.0311      0.0370      32119       0.0000        4        4    yes
(juyetupp AND elecblwi) OR pdsqabzs       0.0105    0.0176      95411     0.0001      0.1700      0.1782       5882       0.0001     1226     1226    yes
(juyetupp AND elecblwi) OR (pdsqa...      0.0189    0.0253      53042     0.0000      0.0675      0.0720      14821       0.0000        9        9    yes
juyetupp AND (elecblwi OR pdsqabz...      0.8192    1.3967       1221     0.0000      0.0688      0.0757      14532       0.0000       12       12    yes
juyetupp OR elecblwi OR pdsqabzs ...      0.0937    0.1026      10674     0.0025      0.8666      0.8927       1154       0.0025     6004     6004    yes
(juyetupp OR elecblwi) AND NOT (p...      0.8151    0.8445       1227     0.0008      0.4119      0.4317       2428       0.0006     2447     2447    yes
```

With `--gpu_copy` on a Tesla T4 (real `cudaMemcpy` D2H instead of a
same-host copy; same corpus as above) -- note the GPU copy's higher, more
uniform floor (fixed launch/sync overhead) compared to the same-host
`*Copy` column:

```bash
PATH=/usr/local/bin:$PATH ./run.sh --gpu_copy
```

```
Index build time: pisa=46.89 ms, xapian=514.05 ms

Query                                   PISA p50  PISA p95   PISA QPS  PISA Copy PISA GPUCopy  Xapian p50  Xapian p95 Xapian QPS  Xapian Copy Xapian GPUCopy    PISA#  Xapian#  Match
                                            (ms)      (ms)               p50(ms)      p50(ms)        (ms)        (ms)                 p50(ms)        p50(ms)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
juyetu                                    0.0001    0.0004   11363636     0.0000       0.0064      0.0174      0.0308      57386       0.0000         0.0063      191      191    yes
juyetu AND ppelec                         0.0009    0.0020    1169591     0.0000       0.0062      0.0053      0.0099     190078       0.0000         0.0063        4        4    yes
(juyetu AND ppelec) OR blwipd             0.0014    0.0028     698812     0.0000       0.0065      0.0260      0.0297      38442       0.0000         0.0065      198      198    yes
(juyetu AND ppelec) OR (blwipd AN...      0.0016    0.0038     633714     0.0000       0.0064      0.0131      0.0301      76208       0.0000         0.0063        6        6    yes
juyetu AND (ppelec OR blwipd) AND...      0.0841    0.0882      11889     0.0000       0.0064      0.0109      0.0171      91684       0.0000         0.0063        5        5    yes
juyetu OR ppelec OR blwipd OR sqa...      0.0078    0.0140     128634     0.0001       0.0073      0.1237      0.1317       8085       0.0001         0.0072      944      944    yes
(juyetu OR ppelec) AND NOT (blwip...      0.0772    0.0862      12960     0.0001       0.0065      0.0574      0.0630      17412       0.0001         0.0067      359      359    yes
```


(default settings -- `vocab_card` large relative to key size, so `Key#`
almost always equals `PISA#`/`Xapian#`):

```bash
PATH=/usr/local/bin:$PATH ./run.sh
```

```
=== Phrase-as-key benchmark (num_vocab_per_key=3, num_keys_per_doc=5) ===
Index build time: key_engine=34.18 ms, pisa=37.15 ms, xapian=388.75 ms

Key                          Key p50   Key p95   Key QPS  Key Copy  PISA p50  PISA p95  PISA QPS PISA Copy Xapian p50 Xapian p95 Xapian QPS XapianCopy   Key#   PISA#  Xapian#
                                (ms)      (ms)             p50(ms)      (ms)      (ms)             p50(ms)       (ms)       (ms)               p50(ms)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
dekrbf ybleyr nzdxny          0.0001    0.0001  14925373    0.0000    0.0007    0.0018   1533742    0.0000     0.0052     0.0267     191351     0.0000      1       1        1
pwaday sokbrl dbavrx          0.0001    0.0001  14084507    0.0000    0.0010    0.0018   1021450    0.0000     0.0054     0.0080     185874     0.0000      1       1        1
ejomfy dycuec ixtlcs          0.0001    0.0001  14705882    0.0000    0.0007    0.0013   1335113    0.0000     0.0055     0.0081     181719     0.0000      1       1        1
punqhn ltlfdj bmweae          0.0001    0.0001  14705882    0.0000    0.0007    0.0013   1492537    0.0000     0.0052     0.0075     193798     0.0000      1       1        1
lmvprf ourwio ydohph          0.0001    0.0001  14084507    0.0000    0.0007    0.0013   1340483    0.0000     0.0055     0.0074     182448     0.0000      1       1        1
```

Phrase-as-key, same corpus size but a much smaller vocabulary (50 terms) and
more keys/doc (10) -- showing `Key#` diverging well below `PISA#`/`Xapian#`
once term reuse across keys becomes likely, and `PISA Copy`/`XapianCopy`
growing as PISA#/Xapian# (result size) grows into the thousands:

```bash
PATH=/usr/local/bin:$PATH ./run.sh --vocab_card=50 --num_keys_per_doc=10
```

```
=== Phrase-as-key benchmark (num_vocab_per_key=3, num_keys_per_doc=10) ===
Index build time: key_engine=51.55 ms, pisa=56.57 ms, xapian=370.34 ms

Key                          Key p50   Key p95   Key QPS  Key Copy  PISA p50  PISA p95  PISA QPS PISA Copy Xapian p50 Xapian p95 Xapian QPS XapianCopy   Key#   PISA#  Xapian#
                                (ms)      (ms)             p50(ms)      (ms)      (ms)             p50(ms)       (ms)       (ms)               p50(ms)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
cmaqim krppld lpoukj          0.0001    0.0001  13888889    0.0000    0.1563    0.1680      6399    0.0002     0.5935     0.6115       1685     0.0002      2    1709     1709
epbuna cmaqim vvpljp          0.0001    0.0001  14925373    0.0000    0.1560    0.1640      6411    0.0002     0.5927     0.6212       1687     0.0002      3    1738     1738
geivwb slmldq cmpcdd          0.0001    0.0001  13888889    0.0000    0.1603    0.1727      6239    0.0002     0.6092     0.6283       1641     0.0002      2    1777     1777
clmsmi vvpljp hyxdfu          0.0001    0.0001  14925373    0.0000    0.1581    0.1650      6326    0.0002     0.5980     0.6171       1672     0.0002      1    1770     1770
qfcfme krppld iwfcch          0.0001    0.0001  14492754    0.0000    0.1578    0.1755      6337    0.0002     0.6028     0.6186       1659     0.0002      1    1783     1783
```

### Phrase-as-key benchmark: design

Some use cases treat a fixed-size tuple of terms (e.g. `"vocA vocB vocC"`) as
one atomic lookup key, not three independent terms to AND together. This
section compares three ways of answering that:

- [`key_engine.h`](key_engine.h) — `KeyEngine`: a plain
  `unordered_map<string, vector<docId>>` keyed by the *entire* key string.
  No tokenization, no posting-list merge -- a single O(1) average-case hash
  lookup per query.
- `PisaEngine`/`XapianEngine` (same engines as above) answering the
  equivalent classic boolean query `vocA AND vocB AND vocC`, i.e. decomposing
  the key into its terms and intersecting their posting lists.

These are **not** semantically equivalent, by design: `KeyEngine` only
matches docs that contain that exact key (the terms together, as generated),
whereas the AND query matches any doc containing all three terms anywhere --
even if they come from different keys in that doc. So `Key#` is expected to
be `<= PISA#`/`Xapian#`, and the gap widens as `vocab_card` shrinks relative
to `num_vocab_per_key * num_keys_per_doc` (more chance of a doc incidentally
containing all 3 terms outside of the generated key itself).

Keys for the benchmark queries are the first 5 distinct keys encountered in
the generated corpus (guaranteed to exist).

### Takeaways (from the runs above, not a general claim)

- **Copy-out cost is small but not zero, and scales with result size, not
  query complexity**: in the default CPU-only runs, copying a handful of
  matching doc ids is unmeasurably fast (`0.0000 ms`), but copying hundreds
  of thousands of doc ids (e.g. the wide-OR query at 200k docs, or the AND
  queries at `vocab_card=50`) costs up to a few tenths of a millisecond --
  occasionally comparable to or larger than the lookup itself for PISA's
  fastest queries. With `--gpu_copy` on an actual GPU, the real
  `cudaMemcpy` D2H transfer is slower still at these result sizes (~0.3-0.5
  ms for 200k-400k ints vs ~0.03-0.12 ms for the same-host copy), and has a
  non-trivial fixed floor (~0.003-0.008 ms) from kernel launch/sync
  overhead even for tiny results -- both costs worth tracking separately
  from lookup/merge compute time, especially for any design that would put
  the index in device memory.
- **Phrase-as-key wins by orders of magnitude when it's the right model**:
  `KeyEngine`'s single hash lookup runs in ~0.0001 ms regardless of corpus
  size/cardinality, versus PISA/Xapian needing to intersect/union real
  posting lists for the equivalent AND query -- a gap that grows as those
  posting lists get longer (e.g. with a small vocabulary, PISA/Xapian's AND
  queries took >0.1-0.6 ms per the second example above, ~1000-5000x
  slower). The tradeoff is semantic, not just a performance knob: `KeyEngine`
  can only answer "does this exact key exist", not "find all docs containing
  these terms in any combination" -- use it only when the former is really
  the question being asked.
- **Index build**: PISA-style in-memory `vector<vector<int>>` postings build
  roughly 10x faster than Xapian's `WritableDatabase::add_document`, which
  does more per-term bookkeeping (positions, term-frequency stats,
  on-disk-ready structures) even in `DB_BACKEND_INMEMORY` mode.
- **Query latency**: no consistent winner across query shapes. PISA-style
  wins on the simplest queries (pure array ops, no object overhead) and on
  wide flat ORs; Xapian wins on queries with multiple AND branches, likely
  because `Xapian::Enquire`'s matcher applies term-frequency-driven
  optimizations (e.g. evaluating rarer terms first) that our naive AST
  evaluator doesn't.
- **Both scale similarly** as corpus size/cardinality grow 10x (20k->200k
  docs), and both agree on every result set across all tested configurations
  and query shapes.
- Take absolute numbers with a grain of salt: this is an in-memory,
  single-query microbenchmark on a synthetic corpus, not a production
  workload. Re-run `./run.sh` with parameters matching your real data shape
  before drawing conclusions for a specific use case.
