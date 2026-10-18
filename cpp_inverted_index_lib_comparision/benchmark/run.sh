#!/bin/bash

rm -f benchmark
make benchmark
./benchmark "$@"
