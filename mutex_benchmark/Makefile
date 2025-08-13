CFLAGS := -pthread -O3 -std=c++17

all: mutex_benchmark

mutex_benchmark: main.cpp
	g++ -o $@ $^ $(CFLAGS)

clean:
	rm -f mutex_benchmark
