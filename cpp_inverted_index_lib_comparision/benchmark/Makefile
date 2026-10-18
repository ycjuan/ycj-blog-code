CFLAGS := -O3 -std=c++17 $(shell xapian-config --cxxflags)
LDFLAGS := $(shell xapian-config --libs)

all: benchmark

benchmark: main.cpp corpus_generator.h query_ast.h pisa_engine.h xapian_engine.h
	g++ -o $@ main.cpp $(CFLAGS) $(LDFLAGS)

clean:
	rm -f benchmark
