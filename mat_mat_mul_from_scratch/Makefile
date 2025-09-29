CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_89 -O3 -lcublas

all: mat_mat_mul_from_scratch

mat_mat_mul_from_scratch: main.cu data.cu method_cpu.cu method_cublus.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f mat_mat_mul_from_scratch
