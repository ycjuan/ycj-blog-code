CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_89 -O3

all: ram_disk_to_gpu_benchmark

ram_disk_to_gpu_benchmark: main.cu
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f ram_disk_to_gpu_benchmark
