# CUDA support (real cudaMalloc + cudaMemcpy D2H GPU-copy columns) is
# optional at build time: only enabled when nvcc is found on PATH. Without
# it, the benchmark still builds and runs fine; it just omits the GPU-copy
# columns. When enabled, the GPU copy is always measured (no runtime flag).
NVCC := $(shell command -v nvcc 2>/dev/null)
CUDA_PATH ?= /usr/local/cuda

CFLAGS := -O3 -std=c++17 $(shell xapian-config --cxxflags)
LDFLAGS := $(shell xapian-config --libs)

ifneq ($(NVCC),)
CFLAGS += -DHAVE_CUDA -I$(CUDA_PATH)/include
LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart
CUDA_OBJ := cuda_copy.o
endif

all: benchmark

benchmark: main.cpp corpus_generator.h query_ast.h pisa_engine.h xapian_engine.h key_engine.h cuda_copy.h $(CUDA_OBJ)
	g++ -o $@ main.cpp $(CUDA_OBJ) $(CFLAGS) $(LDFLAGS)

cuda_copy.o: cuda_copy.cu cuda_copy.h
	$(NVCC) -O3 -std=c++17 -c cuda_copy.cu -o cuda_copy.o

clean:
	rm -f benchmark cuda_copy.o
