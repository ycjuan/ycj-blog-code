CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_89 -O3 -lcublas -lcusparse -lcurand

all: example

example: simpleTensorCoreGEMM.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

sddmm: main.cu baseline.cu cusparse.cu tensor_simple.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

sddmm_test: test.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

sddmm_csr_example: sddmm_csr_example.c
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f sddmm_csr_example sddmm sddmm_test example
