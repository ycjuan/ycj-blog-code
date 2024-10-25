CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_89 -O3

all: test_data_synthesizer main

test_data_synthesizer: test_data_synthesizer.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

main: main.cu topk_with_bucket_sort.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f test_data_synthesizer main
