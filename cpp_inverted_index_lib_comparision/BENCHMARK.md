# PISA-style vs Xapian benchmark

Generates a synthetic corpus with configurable size/cardinality, builds both
a hand-rolled PISA-style index (`pisa_engine.h`) and a real
[Xapian](https://xapian.org) index (`xapian_engine.h`) over the same
documents, runs an identical set of boolean queries against both (via a
shared query AST, see below), and reports result counts + latency side by
side. A second section benchmarks a "phrase-as-key" scenario against both
libraries plus a bespoke hash-map engine (see below).

## Corpus parameters

| Flag | Short | Meaning | Default |
|---|---|---|---|
| `--num_docs` | `-d` | number of documents | 20000 |
| `--num_vocab_per_doc` | `-v` | terms sampled (with replacement) into each doc | 20 |
| `--num_char_per_vocab` | `-l` | character length of each distinct vocabulary term | 6 |
| `--vocab_card` | `-k` | vocabulary cardinality (# of distinct terms overall) | 2000 |
| `--reps` | `-r` | repetitions per query; reports p50/p95 latency across them | 20 |
| `--num_vocab_per_key` | `-p` | terms per key, for the phrase-as-key benchmark | 3 |
| `--num_keys_per_doc` | `-n` | keys per doc, for the phrase-as-key benchmark | 5 |

Long form takes `--flag=value`; short form takes `-f value` (e.g.
`-d 200000 -v 30`). Run `./benchmark --help` for the full usage message.

Generation is seeded (`seed=42`) for reproducibility. Each doc is a
space-joined string of `num_vocab_per_doc` terms drawn uniformly from a fixed
vocabulary of `vocab_card` distinct, `num_char_per_vocab`-character terms.
The phrase-as-key benchmark uses a separate corpus (seeded independently,
`seed=43`) that reuses `num_docs`/`num_char_per_vocab`/`vocab_card` but
generates docs as `num_keys_per_doc` multi-term keys instead (see below).

## Design: one query AST, two evaluators

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

Phrase-as-key, 20,000 docs, 5 keys/doc, 3 terms/key, 2,000-term vocabulary
(default settings -- `vocab_card` large relative to key size, so `Key#`
almost always equals `PISA#`/`Xapian#`):

```bash
PATH=/usr/local/bin:$PATH ./run.sh
```

```
=== Phrase-as-key benchmark (num_vocab_per_key=3, num_keys_per_doc=5) ===
Index build time: key_engine=42.35 ms, pisa=37.47 ms, xapian=401.90 ms

Key                          Key p50   Key p95   Key QPS  PISA p50  PISA p95  PISA QPS Xapian p50 Xapian p95 Xapian QPS   Key#   PISA#  Xapian#
                                (ms)      (ms)                (ms)      (ms)                 (ms)       (ms)                                   
-----------------------------------------------------------------------------------------------------------------------------------------------------
dekrbf ybleyr nzdxny          0.0001    0.0001  15151515    0.0007    0.0018   1347709     0.0051     0.0092     195618      1       1        1
pwaday sokbrl dbavrx          0.0001    0.0001  14492754    0.0011    0.0021    944287     0.0054     0.0080     184026      1       1        1
ejomfy dycuec ixtlcs          0.0001    0.0001  14925373    0.0008    0.0015   1300390     0.0055     0.0081     180538      1       1        1
punqhn ltlfdj bmweae          0.0001    0.0001  14285714    0.0007    0.0017   1428571     0.0052     0.0072     192271      1       1        1
lmvprf ourwio ydohph          0.0001    0.0001  14925373    0.0008    0.0017   1248439     0.0055     0.0075     181686      1       1        1
```

Phrase-as-key, same corpus size but a much smaller vocabulary (50 terms) and
more keys/doc (10) -- showing `Key#` diverging well below `PISA#`/`Xapian#`
once term reuse across keys becomes likely:

```bash
PATH=/usr/local/bin:$PATH ./run.sh --vocab_card=50 --num_keys_per_doc=10
```

```
=== Phrase-as-key benchmark (num_vocab_per_key=3, num_keys_per_doc=10) ===
Index build time: key_engine=66.15 ms, pisa=56.73 ms, xapian=404.50 ms

Key                          Key p50   Key p95   Key QPS  PISA p50  PISA p95  PISA QPS Xapian p50 Xapian p95 Xapian QPS   Key#   PISA#  Xapian#
                                (ms)      (ms)                (ms)      (ms)                 (ms)       (ms)                                   
-----------------------------------------------------------------------------------------------------------------------------------------------------
cmaqim krppld lpoukj          0.0001    0.0001  14705882    0.1701    0.1895      5880     0.5917     0.6180       1690      2    1709     1709
epbuna cmaqim vvpljp          0.0001    0.0001  14705882    0.1683    0.1741      5940     0.5909     0.6121       1692      3    1738     1738
geivwb slmldq cmpcdd          0.0001    0.0001  14492754    0.1722    0.1965      5807     0.6048     0.6263       1653      2    1777     1777
clmsmi vvpljp hyxdfu          0.0001    0.0001  14705882    0.1727    0.1826      5789     0.5987     0.6370       1670      1    1770     1770
qfcfme krppld iwfcch          0.0001    0.0001  11904762    0.1710    0.1794      5849     0.6000     0.6287       1667      1    1783     1783
```

## Phrase-as-key benchmark: design

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

## Takeaways (from the runs above, not a general claim)

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
