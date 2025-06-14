CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_89 -O3 -lcublas

all: col_encoder

col_encoder: main.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f col_encoder
