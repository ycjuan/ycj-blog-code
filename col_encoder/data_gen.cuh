#ifndef DATA_GEN_CUH
#define DATA_GEN_CUH

#include <vector>
#include <random>
#include <omp.h>

#include "data_struct.cuh"

std::vector<std::vector<std::vector<float>>> genRandColEncData(int numRows, int numFields, int embDimPerField)
{
    using namespace std;

    // -----------------
    // Prepare random number generators
    int numThreads = omp_get_max_threads();
    random_device rd;
    vector<default_random_engine> generators;
    for (int t = 0; t < numThreads; t++)
    {
        generators.emplace_back(rd());
    }

    // -----------------
    // Generate random data
    vector<vector<vector<float>>> data(numRows, vector<vector<float>>(numFields, vector<float>(embDimPerField)));

    #pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < numRows; ++i)
    {
        uniform_real_distribution<float> distribution(0, sqrt(1.0f / embDimPerField));
        for (int j = 0; j < numFields; ++j)
        {
            for (int k = 0; k < embDimPerField; ++k)
            {
                data[i][j][k] = distribution(generators[omp_get_thread_num()]);
            }
        }
    }

    return data;
}

std::vector<ScoringTask> genRandScoringTasks(int numReqs, int numToScore, int numDocs)
{
    using namespace std;

    default_random_engine generator;
    uniform_int_distribution<int> distribution(0, numDocs - 1);

    vector<ScoringTask> tasks(numReqs * numToScore);
    for (int reqIdx = 0; reqIdx < numReqs; ++reqIdx)
    {
        for (int docIdx = 0; docIdx < numToScore; ++docIdx)
        {
            int taskIdx = reqIdx * numToScore + docIdx;
            tasks[taskIdx].reqIdx = reqIdx;
            tasks[taskIdx].docIdx = distribution(generator);
            tasks[taskIdx].result = 0.0f;
        }
    }

    return tasks;
}

#endif // DATA_GEN_CUH