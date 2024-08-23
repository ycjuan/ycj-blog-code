CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_89 -O3

all: bf16_vs_fp32

bf16_vs_fp32: main.cu util.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f bf16_vs_fp32
