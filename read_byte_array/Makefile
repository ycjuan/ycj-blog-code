CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_70 -O3

all: byte_array

byte_array: byte_array.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f byte_array
