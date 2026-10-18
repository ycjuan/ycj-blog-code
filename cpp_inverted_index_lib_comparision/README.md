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
| `--gpu_copy` | `-g` | also measure a real `cudaMalloc`+`cudaMemcpy` D2H copy per query | off |

Long form takes `--flag=value`; short form takes `-f value` (e.g.
`-d 200000 -k 5000`). Run `./benchmark --help` for the full usage message.

Generation is seeded (`seed=42`) for reproducibility. A fixed pool of
`phrase_card` distinct *seed phrases* is generated up front (each a
`num_vocab_per_phrase`-term tuple, each term `num_char_per_vocab`
characters). Every doc is then built by sampling `num_phrases_per_doc` of
those *same* seed phrases (with replacement) -- so a given phrase recurs
across many docs whenever `num_docs` is large relative to `phrase_card`.
This single corpus backs both halves of the benchmark: PISA/Xapian see each
doc as the flat bag of all terms across its sampled phrases, while
`KeyEngine` sees each doc as its list of sampled phrase strings (see
below). Reusing the same seed-phrase pool across docs is what makes the
phrase-as-key benchmark meaningful -- a phrase has to actually recur to have
anything interesting to look up (an earlier version of this benchmark
sampled each phrase's terms independently per doc, so phrases almost never
repeated and `Key#` was ~1 for nearly every query).


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

20,000 docs, phrase_card=2000, 3 terms/phrase, 5 phrases/doc:

```
Index build time: pisa=59.16 ms, xapian=412.38 ms, key_engine=8.21 ms

Query                                   PISA p50  PISA p95   PISA QPS  PISA Copy  Xapian p50  Xapian p95 Xapian QPS  Xapian Copy    PISA#  Xapian#  Match
                                            (ms)      (ms)               p50(ms)        (ms)        (ms)                 p50(ms)
---------------------------------------------------------------------------------------------------------------------------------------------------------
juyetu                                    0.0001    0.0002   12987013     0.0000      0.0052      0.0162     193461       0.0000       52       52    yes
juyetu AND ppelec                         0.0004    0.0007    2347418     0.0000      0.0086      0.0094     115701       0.0000       52       52    yes
(juyetu AND ppelec) OR blwipd             0.0008    0.0009    1303781     0.0000      0.0103      0.0111      96665       0.0000       52       52    yes
(juyetu AND ppelec) OR (blwipd AN...      0.0010    0.0013    1034126     0.0000      0.0109      0.0117      91971       0.0000       52       52    yes
juyetu AND (ppelec OR blwipd) AND...      0.0875    0.0966      11434     0.0000      0.0133      0.0143      75171       0.0000       52       52    yes
juyetu OR ppelec OR blwipd OR sqa...      0.0018    0.0028     559284     0.0000      0.0171      0.0208      58353       0.0000       96       96    yes
(juyetu OR ppelec) AND NOT (blwip...      0.0647    0.0691      15450     0.0000      0.0117      0.0126      85463       0.0000       52       52    yes
```

200,000 docs, phrase_card=5000, 8-char terms:

```bash
PATH=/usr/local/bin:$PATH ./run.sh -d 200000 -k 5000 -l 8
```

```
Index build time: pisa=451.77 ms, xapian=4611.00 ms, key_engine=113.29 ms

Query                                   PISA p50  PISA p95   PISA QPS  PISA Copy  Xapian p50  Xapian p95 Xapian QPS  Xapian Copy    PISA#  Xapian#  Match
                                            (ms)      (ms)               p50(ms)        (ms)        (ms)                 p50(ms)
---------------------------------------------------------------------------------------------------------------------------------------------------------
juyetupp                                  0.0001    0.0002   13157895     0.0000      0.0170      0.0302      58696       0.0000      186      186    yes
juyetupp AND elecblwi                     0.0008    0.0012    1210654     0.0000      0.0244      0.0260      40984       0.0000      186      186    yes
(juyetupp AND elecblwi) OR pdsqabzs       0.0015    0.0023     661813     0.0000      0.0288      0.0386      34685       0.0000      186      186    yes
(juyetupp AND elecblwi) OR (pdsqa...      0.0020    0.0035     497265     0.0000      0.0288      0.0310      34771       0.0000      186      186    yes
juyetupp AND (elecblwi OR pdsqabz...      0.9018    0.9321       1109     0.0000      0.0380      0.0413      26327       0.0000      186      186    yes
juyetupp OR elecblwi OR pdsqabzs ...      0.0038    0.0071     263783     0.0001      0.0587      0.0635      17042       0.0001      379      379    yes
(juyetupp OR elecblwi) AND NOT (p...      0.6762    0.6977       1479     0.0000      0.0320      0.0360      31297       0.0000      186      186    yes
```

With `--gpu_copy` on a Tesla T4 (real `cudaMemcpy` D2H instead of a
same-host copy; 20,000-doc corpus) -- note the GPU copy's higher, more
uniform floor (fixed launch/sync overhead) compared to the same-host
`*Copy` column:

```bash
PATH=/usr/local/bin:$PATH ./run.sh --gpu_copy
```

```
Index build time: pisa=58.0 ms, xapian=410.3 ms, key_engine=8.1 ms

Query                                   PISA p50  PISA p95   PISA QPS  PISA Copy PISA GPUCopy  Xapian p50  Xapian p95 Xapian QPS  Xapian Copy Xapian GPUCopy    PISA#  Xapian#  Match
                                            (ms)      (ms)               p50(ms)      p50(ms)        (ms)        (ms)                 p50(ms)        p50(ms)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
juyetu                                    0.0001    0.0002   13157895     0.0000       0.0068      0.0022      0.0091     456621       0.0000         0.0067       14       14    yes
juyetu AND ppelec                         0.0003    0.0007    3584229     0.0000       0.0067      0.0047      0.0274     211595       0.0000         0.0067       14       14    yes
(juyetu AND ppelec) OR blwipd             0.0005    0.0008    2053388     0.0000       0.0067      0.0057      0.0075     175346       0.0000         0.0064       14       14    yes
(juyetu AND ppelec) OR (blwipd AN...      0.0006    0.0010    1779359     0.0000       0.0067      0.0066      0.0080     151837       0.0000         0.0064       14       14    yes
```

Phrase-as-key, 20,000 docs, 5 phrases/doc, 3 terms/phrase, phrase_card=2000
(default settings). Because each seed phrase's terms are generated fresh
and never shared with any other phrase, a phrase lookup and the equivalent
AND query always match the same doc set here (`Key#` == `PISA#` ==
`Xapian#`) -- see the design note below for when that can diverge:

```bash
PATH=/usr/local/bin:$PATH ./run.sh
```

```
=== Phrase-as-key benchmark (num_vocab_per_phrase=3, num_phrases_per_doc=5) ===

Key                          Key p50   Key p95   Key QPS  Key Copy  PISA p50  PISA p95  PISA QPS PISA Copy Xapian p50 Xapian p95 Xapian QPS XapianCopy   Key#   PISA#  Xapian#
                                (ms)      (ms)             p50(ms)      (ms)      (ms)             p50(ms)       (ms)       (ms)               p50(ms)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
dhovhb bubpdo blthmm          0.0001    0.0001  13698630    0.0000    0.0008    0.0010   1215067    0.0000     0.0099     0.0107     101092     0.0000     53      53       53
lekece rrslsz sfaiyz          0.0001    0.0001  13333333    0.0000    0.0008    0.0009   1225490    0.0000     0.0096     0.0102     104679     0.0000     52      52       52
pswert lzvuxa dpnshv          0.0001    0.0001  11363636    0.0000    0.0008    0.0009   1221001    0.0000     0.0100     0.0103     100412     0.0000     55      55       55
aavfss suupby jqdxww          0.0001    0.0001  12987013    0.0000    0.0008    0.0010   1216545    0.0000     0.0103     0.0136      97087     0.0000     58      58       58
nagyec vudxmy ywehzi          0.0001    0.0001  14285714    0.0000    0.0007    0.0008   1336898    0.0000     0.0086     0.0093     115727     0.0000     42      42       42
```

Note that `Key#`/`PISA#`/`Xapian#` all grow with `num_docs / phrase_card`
(here ~50-58, close to the expected `20000 * 5 / 2000 = 50`) -- each seed
phrase recurs across roughly that many docs, which is the whole point of
sampling from a fixed phrase pool instead of generating independent random
terms per doc.

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
even if they come from different keys in that doc. So in general `Key#` is
expected to be `<= PISA#`/`Xapian#`. In this benchmark's corpus, every seed
phrase's terms are generated fresh and distinct from every other phrase's
terms (see "Corpus parameters" above), so that gap never actually opens up
here -- `Key#` always equals `PISA#`/`Xapian#` in the example output.
The gap would reappear with a corpus where individual terms (not whole
phrases) are reused across different phrases, making it possible for a doc
to incidentally contain all of a key's terms without containing that exact
phrase.

Keys for the benchmark queries are the first 5 distinct phrases encountered
in the generated corpus (guaranteed to exist).

### Takeaways (from the runs above, not a general claim)

- **Fixing a phrase-reuse bug**: an earlier version of this benchmark
  generated each phrase's terms independently per doc from a shared term
  vocabulary, so the same exact phrase almost never recurred across docs
  (`Key#` was ~1 for nearly every query, regardless of corpus size). Sampling
  `num_phrases_per_doc` phrases per doc from a *fixed* pool of `phrase_card`
  seed phrases fixes this: `Key#` now scales with `num_docs / phrase_card`
  as expected (see the phrase-as-key example above).
- **Copy-out cost is small but not zero, and scales with result size, not
  query complexity**: in the default CPU-only runs, copying a handful of
  matching doc ids is unmeasurably fast (`0.0000 ms`), but copying hundreds
  of thousands of doc ids costs up to a few tenths of a millisecond --
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
  posting lists get longer (e.g. at 200k docs, PISA/Xapian's AND queries
  took up to ~1 ms, ~10000x slower). The tradeoff is semantic, not just a
  performance knob: `KeyEngine` can only answer "does this exact key exist",
  not "find all docs containing these terms in any combination" -- use it
  only when the former is really the question being asked.
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
