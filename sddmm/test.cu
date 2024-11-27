#include <string>
#include <stdexcept>
#include <iostream>
#include <random>
#include <sstream>
#include <cublas_v2.h>
#include <type_traits>
#include <algorithm>
#include <cassert>

#include "util.cuh"
#include "common.cuh"
#include "methods.cuh"

using namespace std;

void testCooToCsr()
{
    Data data;
    data.numDocs = 6;
    data.numReqs = 14;
    data.numPairsToScore = 9;
    vector<Pair> v_pairsToScore = {
        {0, 0, 0.0},
        {0, 7, 0.1},
        {2, 1, 0.2},
        {2, 6, 0.3},
        {2, 10, 0.4},
        {5, 3, 0.5},
        {5, 9, 0.6},
        {5, 12, 0.7},
        {5, 13, 0.8}
    };

    shuffle(v_pairsToScore.begin(), v_pairsToScore.end(), default_random_engine(0));

    vector<int> v_offsets(data.numDocs + 1, 0);
    vector<int> v_columns(data.numPairsToScore, 0);
    vector<float> v_values(data.numPairsToScore, 0.0);

    coo2Csr(data, v_offsets.data(), v_columns.data(), v_values.data());

    assert(v_offsets == vector<int>({0, 2, 2, 5, 5, 5, 9}));
    assert(v_columns == vector<int>({0, 7, 1, 6, 10, 3, 9, 12, 13}));
    assert(v_values == vector<float>({0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}));
}

int main()
{

    return 0;
}