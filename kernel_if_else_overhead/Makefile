CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_89 -O3 -lcublas

all: main

main: main.cu kernel.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f main