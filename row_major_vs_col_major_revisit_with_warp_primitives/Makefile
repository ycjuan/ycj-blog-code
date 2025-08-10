CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_75 -O3

all: row_vs_col_v2

row_vs_col_v2: main.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f row_vs_col_v2
