CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
CFLAGS       := -O3 -std=c++17

all: std_vector_at

std_vector_at: main.cpp
	g++ $(INC) -o $@ $^ $(CFLAGS)

clean:
	rm -f std_vector_at
