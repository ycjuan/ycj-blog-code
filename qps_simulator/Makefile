CFLAGS := -pthread -O3 -std=c++17

all: qps_simulator

qps_simulator: main.cpp
	g++ -o $@ $^ $(CFLAGS)

clean:
	rm -f qps_simulator
