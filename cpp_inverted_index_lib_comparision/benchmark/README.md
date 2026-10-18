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
| `--reps` | `-r` | repetitions per query; reported latency is the median | 5 |

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
# or with custom corpus parameters:
PATH=/usr/local/bin:$PATH ./run.sh --num_docs=200000 --num_vocab_per_doc=30 --vocab_card=5000 --num_char_per_vocab=8 --reps=7
```

## Example results

20,000 docs, 20 terms/doc, 2,000-term vocabulary:

```
Index build time: pisa=44.48 ms, xapian=489.38 ms

Query                                                           PISA(us)  Xapian(us)    PISA count  Xapian count     Match
--------------------------------------------------------------------------------------------------------------------------
juyetu                                                               0.1        18.3           191           191       yes
juyetu AND ppelec                                                    1.4         6.4             4             4       yes
(juyetu AND ppelec) OR blwipd                                        1.6        26.0           198           198       yes
(juyetu AND ppelec) OR (blwipd AND sqabzs)                           2.3        13.4             6             6       yes
juyetu AND (ppelec OR blwipd) AND NOT sqabzs                        85.1        12.6             5             5       yes
juyetu OR ppelec OR blwipd OR sqabzs OR vyfaez                      11.3       127.3           944           944       yes
(juyetu OR ppelec) AND NOT (blwipd AND sqabzs)                      80.2        57.4           359           359       yes
```

200,000 docs, 30 terms/doc, 5,000-term vocabulary:

```
Index build time: pisa=661.68 ms, xapian=8201.16 ms

Query                                                           PISA(us)  Xapian(us)    PISA count  Xapian count     Match
--------------------------------------------------------------------------------------------------------------------------
juyetupp                                                             0.2       117.6          1194          1194       yes
juyetupp AND elecblwi                                                9.8        32.6             4             4       yes
(juyetupp AND elecblwi) OR pdsqabzs                                 12.7       170.7          1226          1226       yes
(juyetupp AND elecblwi) OR (pdsqabzs AND vyfaezeq)                  23.6        68.8             9             9       yes
juyetupp AND (elecblwi OR pdsqabzs) AND NOT vyfaezeq               931.4        69.9            12            12       yes
juyetupp OR elecblwi OR pdsqabzs OR vyfaezeq OR hpnalahn            85.9       868.9          6004          6004       yes
(juyetupp OR elecblwi) AND NOT (pdsqabzs AND vyfaezeq)             940.1       413.1          2447          2447       yes
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
