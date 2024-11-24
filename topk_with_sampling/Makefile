CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_89 -O3

all: topk

topk: main.cu topk_baseline.cu topk_with_bucket_sort.cu util.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f topk
