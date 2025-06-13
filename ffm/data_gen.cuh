#ifndef DATA_GEN_CUH
#define DATA_GEN_CUH

#include <vector>
#include <random>

#include "data_struct.cuh"

std::vector<std::vector<std::vector<float>>> genRandFFMData(int numRows, int numFields, int embDimPerField)
{
    using namespace std;

    default_random_engine generator;
    uniform_real_distribution<float> distribution(0, sqrt(1.0f / embDimPerField));

    vector<vector<vector<float>>> data(numRows, vector<vector<float>>(numFields, vector<float>(embDimPerField)));

    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < numFields; ++j)
        {
            for (int k = 0; k < embDimPerField; ++k)
            {
                data[i][j][k] = distribution(generator);
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