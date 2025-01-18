CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_89 -O3 -lcublas

all: tensor_quant

tensor_quant: main.cu baseline.cu tensor_simple.cu tensor_unroll.cu tensor_unroll_v2.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f tensor_quant