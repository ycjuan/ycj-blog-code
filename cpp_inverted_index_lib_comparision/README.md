# C++ inverted index library comparison

Two demos built over the same toy corpus and the same boolean queries, to
compare how different C++ libraries handle an inverted index and boolean
(AND/OR/NOT) retrieval.

| | [`pisa/`](pisa) | [`xapian/`](xapian) |
|---|---|---|
| Library | hand-rolled, modeled after [PISA](https://github.com/pisa-engine/pisa)'s core concepts | real [Xapian](https://xapian.org) |
| Term dictionary / posting lists | custom `TermDictionary` + `vector<vector<int>>` | owned internally by Xapian |
| Flat AND / OR | custom `queryAnd`/`queryOr` merge functions | `Xapian::Query::OP_AND` / `OP_OR` |
| Nested boolean (`(A AND B) OR (NOT C)`) | custom `BooleanQueryEngine` (hand-written recursive-descent parser) | native `Xapian::Query` trees, no custom parser needed |
| Ranking | none (not attempted) | explicitly disabled via `Xapian::BoolWeight` (Xapian supports ranking/BM25 natively, we just don't use it here) |
| Dependencies | none (single `g++` call) | requires building/installing libxapian |

## Why two demos?

PISA is primarily designed for large-scale, SIMD-compressed, *ranked* top-k
retrieval (WAND/MaxScore/BlockMax-WAND over disjunctive term lists) -- it
does not ship a general nested-boolean-expression query layer. To support
queries like `(quick AND dog) OR (NOT fox)` on top of PISA's primitives, you
have to write your own recursive evaluator, so `pisa/` demonstrates exactly
that: a minimal from-scratch inverted index plus a hand-written boolean query
parser/evaluator, to show what PISA (or any engine offering only flat
AND/OR primitives) would need glued on top.

Xapian, by contrast, is a general-purpose IR library whose `Xapian::Query`
is natively a *tree* of operators (`OP_AND`, `OP_OR`, `OP_AND_NOT`, `OP_XOR`,
...), so nested boolean expressions are first-class -- no custom parser
required. `xapian/` shows the identical corpus and identical queries
expressed directly as `Xapian::Query` trees for direct comparison.

Both demos produce **identical results** for all queries, confirming the two
approaches are semantically equivalent -- the difference is in how much
boolean-query machinery you have to write yourself.

## Quick summary

- **Want to learn the fundamentals, or need a tiny dependency-free component?**
  See `pisa/` -- shows dictionary + posting lists + merge-based set ops from
  scratch, and how a nested-boolean parser can be layered on top of flat
  AND/OR primitives in ~60 lines.
- **Want pure boolean retrieval in production without writing your own query
  tree evaluator?** See `xapian/` -- a mature, actively maintained C++
  library with native nested boolean queries (and ranking available if
  later needed, unlike PISA).
- **Want large-scale ranked top-k retrieval over huge compressed indexes?**
  Neither demo covers this directly, but PISA (full library, not our toy
  re-implementation) is purpose-built for exactly that; Xapian also supports
  ranked search (BM25 and others) if boolean-only isn't the end goal.

## Running both

```bash
cd pisa && ./run.sh
cd ../xapian && PATH=/usr/local/bin:$PATH ./run.sh
```

## Benchmark

[`benchmark/`](benchmark) runs both engines against a shared, size-configurable
synthetic corpus and an identical set of boolean queries (parsed once into a
shared AST, then evaluated by each engine), reporting result counts and
latency side by side -- see [`benchmark/README.md`](benchmark/README.md) for
methodology and example results.

```bash
cd benchmark && PATH=/usr/local/bin:$PATH ./run.sh --num_docs=200000 --num_vocab_per_doc=30 --vocab_card=5000
```
