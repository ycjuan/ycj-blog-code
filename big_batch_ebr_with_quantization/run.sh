#!/bin/bash

rm -f test_emb_op
make test_emb_op
./test_emb_op

rm -f test_topk
make test_topk
./test_topk

rm -f test_quant_op
make test_quant_op
./test_quant_op
