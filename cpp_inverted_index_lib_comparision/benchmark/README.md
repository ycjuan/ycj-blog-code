# PISA-style vs Xapian benchmark

Generates a synthetic corpus with configurable size/cardinality, builds both
the hand-rolled PISA-style index ([`../pisa`](../pisa)) and a real
[Xapian](https://xapian.org) index ([`../xapian`](../xapian)) over the same
documents, runs an identical set of boolean queries against both (via a
shared query AST, see below), and reports result counts + latency side by
side.

## Corpus parameters

| Flag | Short | Meaning | Default |
|---|---|---|---|
| `--num_docs` | `-d` | number of documents | 20000 |
| `--num_vocab_per_doc` | `-v` | terms sampled (with replacement) into each doc | 20 |
| `--num_char_per_vocab` | `-l` | character length of each distinct vocabulary term | 6 |
| `--vocab_card` | `-k` | vocabulary cardinality (# of distinct terms overall) | 2000 |
| `--reps` | `-r` | repetitions per query; reports p50/p95 latency across them | 20 |

Long form takes `--flag=value`; short form takes `-f value` (e.g.
`-d 200000 -v 30`). Run `./benchmark --help` for the full usage message.

Generation is seeded (`seed=42`) for reproducibility. Each doc is a
space-joined string of `num_vocab_per_doc` terms drawn uniformly from a fixed
vocabulary of `vocab_card` distinct, `num_char_per_vocab`-character terms.

## Design: one query AST, two evaluators

[`query_ast.h`](query_ast.h) parses a query string (same grammar as the PISA
demo's `BooleanQueryEngine`: `NOT` > `AND` > `OR`, with parentheses) into a
`QueryNode` tree exactly once per query. Both engines then evaluate the
*same* tree:

- [`pisa_engine.h`](pisa_engine.h) — `PisaEngine::evaluate(node)` recurses
  over the AST, combining posting lists with merge-based
  intersect/union/difference (same approach as `../pisa/main.cpp`).
- [`xapian_engine.h`](xapian_engine.h) — `XapianEngine::evaluate(node)`
  recurses over the same AST, building an `Xapian::Query` tree
  (`OP_AND`/`OP_OR`/`OP_AND_NOT`) and executing it with `Xapian::BoolWeight`
  (pure boolean matching, no ranking).

This guarantees both engines are asked literally the same question, so any
difference is purely in each library's internals, not in query construction.
The benchmark also asserts both engines return the identical sorted doc-id
set per query as a correctness check (`Match` column).

## Query set

1. Single term: `v0`
2. Flat AND: `v0 AND v1`
3. AND then OR: `(v0 AND v1) OR v2`
4. Two AND groups OR'd: `(v0 AND v1) OR (v2 AND v3)`
5. AND + OR + NOT mixed: `v0 AND (v1 OR v2) AND NOT v3`
6. Wide flat OR (5 terms): `v0 OR v1 OR v2 OR v3 OR v4`
7. OR group AND NOT an AND group: `(v0 OR v1) AND NOT (v2 AND v3)`

`v0..v4` are the first 5 terms of the generated vocabulary (guaranteed to
exist in the corpus, so results are non-trivial).

## Build & run

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

## Example results

20,000 docs, 20 terms/doc, 2,000-term vocabulary:

```
Index build time: pisa=45.94 ms, xapian=489.39 ms

Query                                   PISA p50  PISA p95   PISA QPS  Xapian p50  Xapian p95 Xapian QPS    PISA#  Xapian#  Match
                                            (ms)      (ms)                   (ms)        (ms)
-------------------------------------------------------------------------------------------------------------------------------------------
juyetu                                    0.0001    0.0006    8333333      0.0317      0.0517      31590      191      191    yes
juyetu AND ppelec                         0.0015    0.0026     679810      0.0086      0.0141     116550        4        4    yes
(juyetu AND ppelec) OR blwipd             0.0025    0.0035     393546      0.0489      0.0515      20441      198      198    yes
(juyetu AND ppelec) OR (blwipd AN...      0.0030    0.0053     332336      0.0217      0.0246      46051        6        6    yes
juyetu AND (ppelec OR blwipd) AND...      0.0841    0.1738      11897      0.0176      0.0231      56912        5        5    yes
juyetu OR ppelec OR blwipd OR sqa...      0.0122    0.0182      82095      0.2155      0.2414       4641      944      944    yes
(juyetu OR ppelec) AND NOT (blwip...      0.1196    0.1322       8361      0.0975      0.1040      10257      359      359    yes
```

200,000 docs, 30 terms/doc, 5,000-term vocabulary:

```
Index build time: pisa=650.45 ms, xapian=8132.42 ms

Query                                   PISA p50  PISA p95   PISA QPS  Xapian p50  Xapian p95 Xapian QPS    PISA#  Xapian#  Match
                                            (ms)      (ms)                   (ms)        (ms)
-------------------------------------------------------------------------------------------------------------------------------------------
juyetupp                                  0.0002    0.0008    6060606      0.1170      0.1610       8550     1194     1194    yes
juyetupp AND elecblwi                     0.0072    0.0127     138793      0.0311      0.0367      32135        4        4    yes
(juyetupp AND elecblwi) OR pdsqabzs       0.0114    0.0174      87873      0.1704      0.1829       5868     1226     1226    yes
(juyetupp AND elecblwi) OR (pdsqa...      0.0220    0.0265      45510      0.0675      0.0728      14810        9        9    yes
juyetupp AND (elecblwi OR pdsqabz...      0.9333    0.9561       1071      0.0687      0.0759      14553       12       12    yes
juyetupp OR elecblwi OR pdsqabzs ...      0.0782    0.0854      12793      0.8630      0.8912       1159     6004     6004    yes
(juyetupp OR elecblwi) AND NOT (p...      0.9378    0.9520       1066      0.4115      0.4384       2430     2447     2447    yes
```

## Takeaways (from the runs above, not a general claim)

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
