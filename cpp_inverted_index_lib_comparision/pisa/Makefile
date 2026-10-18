CFLAGS := -O3 -std=c++17

all: pisa_demo

pisa_demo: main.cpp
	g++ -o $@ $^ $(CFLAGS)

clean:
	rm -f pisa_demo
