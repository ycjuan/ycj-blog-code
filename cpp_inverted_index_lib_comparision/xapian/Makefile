CFLAGS := -O3 -std=c++17 $(shell xapian-config --cxxflags)
LDFLAGS := $(shell xapian-config --libs)

all: xapian_demo

xapian_demo: main.cpp
	g++ -o $@ $^ $(CFLAGS) $(LDFLAGS)

clean:
	rm -f xapian_demo
