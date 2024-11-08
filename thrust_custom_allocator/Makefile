CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -Xcompiler -fopenmp --gpu-architecture=sm_89 -O3

all: run_default_thrust_sort run_malloc run_thrust_sort_with_custom_allocator

run_default_thrust_sort: run_default_thrust_sort.cu 
	nvcc $(INC) -o $@ $^ $(CFLAGS)

run_malloc: run_malloc.cu 
	nvcc $(INC) -o $@ $^ $(CFLAGS)

run_thrust_sort_with_custom_allocator: run_thrust_sort_with_custom_allocator.cu 
	nvcc $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f run_default_thrust_sort run_malloc run_thrust_sort_with_custom_allocator