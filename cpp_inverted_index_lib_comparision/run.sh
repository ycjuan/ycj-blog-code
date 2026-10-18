#!/bin/bash
#
# Usage: ./run.sh [OPTIONS]
#
# Builds and runs the benchmark. All arguments are forwarded to the
# benchmark binary; run with -h/--help to see available options, e.g.:
#   ./run.sh --help
#   ./run.sh --num_docs=200000 --num_vocab_per_doc=30 --vocab_card=5000

rm -f benchmark
make benchmark
./benchmark "$@"
