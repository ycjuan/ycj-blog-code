CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_89 -O3 -lcublas

all: is_in

is_in: main.cu baseline.cu binary_search_plus.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f is_in
