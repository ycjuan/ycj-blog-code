# C++ inverted index library comparison

Compares how different C++ approaches handle an inverted index and boolean
(AND/OR/NOT) retrieval over the same synthetic corpus and identical queries.
All code lives in this directory -- see [`BENCHMARK.md`](BENCHMARK.md) for
full methodology and example results.

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
equivalent to the AND query over the same terms (see
[`BENCHMARK.md`](BENCHMARK.md) for why), but is dramatically
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

## Running the benchmark

The benchmark (`main.cpp` + the engine headers) generates a
size-configurable synthetic corpus, builds all engines over it, and runs an
identical set of boolean queries (parsed once into a shared AST, then
evaluated by each engine) plus a separate phrase-as-key benchmark, reporting
result counts and p50/p95 latency side by side -- see
[`BENCHMARK.md`](BENCHMARK.md) for full methodology and example results.

```bash
PATH=/usr/local/bin:$PATH ./run.sh --num_docs=200000 --num_vocab_per_doc=30 --vocab_card=5000
```
