CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_89 -O3 -lcublas

all: test_topk test_emb_op test_quant_op

test_emb_op: test_emb_op.cu emb_data_struct.cu emb_op.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

test_quant_op: test_quant_op.cu quant_data_struct.cu quant_op.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

test_topk: test_topk.cu topk_baseline.cu topk_with_sampling.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f test_topk test_emb_op test_quant_op
