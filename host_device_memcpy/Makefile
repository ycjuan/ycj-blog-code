CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_89 -O3 -lcublas

all: host_device_memcpy

host_device_memcpy: main.cu methods.cu common.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f host_device_memcpy