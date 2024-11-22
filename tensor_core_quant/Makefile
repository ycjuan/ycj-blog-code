CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_89 -O3 -lcublas

all: tensor_quant

tensor_quant: main.cu baseline.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f tensor_quant