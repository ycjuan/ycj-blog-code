CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_89 -O3

all: random_sample

random_sample: main.cu util.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f random_sample
