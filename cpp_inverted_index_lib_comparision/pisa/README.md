# PISA-style inverted index demo

A minimal, self-contained inverted index demo in C++, modeled after the core
ideas used by [PISA](https://github.com/pisa-engine/pisa) (Performant Indexes
and Search for Academia):

- a term -> posting list dictionary (`unordered_map<string, vector<int>>`)
- sorted doc-id posting lists
- merge-based conjunctive (AND) and disjunctive (OR) query processing

This is *not* a build of the real PISA library — PISA depends on ~15 git
submodules (FastPFor, streamvbyte, MaskedVByte, simdcomp, QMX, Boost, GSL,
spdlog, fmt, gumbo-parser via autotools, etc.) and is meant for large-scale,
on-disk, SIMD-compressed indexes. This demo instead shows the same querying
concepts (dictionary + postings + AND/OR merge) in ~150 lines so it builds
with a single `g++` call.

## Build & run

```bash
./run.sh
```

## What it does

1. Tokenizes a handful of toy documents and builds an inverted index, using a
   `TermDictionary` (string -> dense integer `term_id`) so query processing
   never hashes strings — only plain integer intersections/unions.
2. Runs flat AND / OR queries via `InvertedIndex::queryAnd` / `queryOr`.
3. Runs **complex nested boolean queries** (AND / OR / NOT with parentheses)
   via `BooleanQueryEngine`, a small recursive-descent parser + evaluator,
   e.g. `"(quick AND dog) OR (NOT fox)"`, `"quick AND (dog OR fox) AND NOT lazy"`.
   Precedence follows the usual convention: `NOT` > `AND` > `OR`.

## Using the real PISA library

To build the actual PISA library/tools:

```bash
git clone --recursive https://github.com/pisa-engine/pisa.git
cd pisa && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

PISA then provides CLI tools (`parse_collection`, `invert`,
`create_freq_index`, `queries`, ...) for building and querying compressed,
disk-backed inverted indexes over large text collections.
