CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_89 -O3 -lcublas -lcusparse

all: sddmm sddmm_test

sddmm: main.cu baseline.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

sddmm_test: test.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

sddmm_csr_example: sddmm_csr_example.c
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f sddmm_csr_example sddmm sddmm_test