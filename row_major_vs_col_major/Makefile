CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_70 -O3

all: row_vs_col

row_vs_col: row_vs_col.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f row_vs_col
