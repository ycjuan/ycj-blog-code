CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_89 -O3 -lcublas

all: many_cuda_streams

many_cuda_streams: main.cu 
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f many_cuda_streams
