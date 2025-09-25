CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_89 -O3 -lcublas

all: batch_scalibility

batch_scalibility: main.cu data.cu method_cpu.cu method_gpu.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f batch_scalibility
