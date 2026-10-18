# Xapian inverted index demo

Same toy corpus and same boolean queries as [`../pisa/main.cpp`](../pisa/main.cpp),
but built on the real [Xapian](https://xapian.org) C++ library instead of a
hand-rolled inverted index.

Xapian owns the term dictionary, posting lists, and compression internally --
we just call `doc.add_term(...)` to index and build `Xapian::Query` trees to
search. Nested boolean queries (`AND` / `OR` / `AND_NOT`) are natively
supported via `Xapian::Query::OP_AND` / `OP_OR` / `OP_AND_NOT`, so unlike the
PISA demo, no custom `BooleanQueryEngine` parser was needed.

## Prerequisites

Xapian isn't in the Amazon Linux 2023 default repos, so it was built from
source for this demo:

```bash
curl -sL -o xapian-core.tar.xz https://oligarchy.co.uk/xapian/1.4.25/xapian-core-1.4.25.tar.xz
tar xf xapian-core.tar.xz && cd xapian-core-1.4.25
./configure --prefix=/usr/local
make -j4
sudo make install
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/xapian.conf
sudo ldconfig
```

On other distros, prefer a system package, e.g.:

```bash
sudo apt-get install libxapian-dev   # Debian/Ubuntu
sudo dnf install xapian-core-devel   # Fedora
```

## Build & run

```bash
PATH=/usr/local/bin:$PATH ./run.sh
```

(The `PATH` prefix is only needed if `xapian-config` was installed under
`/usr/local/bin` rather than a system path already on `PATH`. Registering
`/usr/local/lib` with `ldconfig` above means `LD_LIBRARY_PATH` isn't needed
at runtime.)

## What it does

1. Builds an in-memory `Xapian::WritableDatabase` (`DB_BACKEND_INMEMORY`) and
   indexes the same 6 toy documents as the PISA demo.
2. Runs the same flat AND / OR queries (`quick AND dog`, `index OR pisa`,
   `quick AND dog AND park`) using `Xapian::Query::OP_AND` / `OP_OR` over a
   `vector<Xapian::Query>`.
3. Sets `Xapian::BoolWeight` so matching is pure boolean set membership, with
   no relevance scoring/ranking applied.
4. Runs the same 4 nested boolean queries as the PISA demo
   (`(quick AND dog) OR (NOT fox)`, `quick AND (dog OR fox) AND NOT lazy`,
   etc.), built directly as `Xapian::Query` trees -- no custom parser.
   Xapian has no standalone `NOT` operator; `NOT x` is expressed as
   `OP_AND_NOT(MatchAll, x)`.

Results match the PISA demo's `BooleanQueryEngine` output exactly.
