#ifndef PRE_GPU_CUH
#define PRE_GPU_CUH

#include "data_struct.cuh"
#include "common.cuh"
#include "topk.cuh"
#include "misc.cuh"
#include "util.cuh"

#include <random>
#include <iostream>
#include <algorithm>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

using namespace std;

vector<vector<ReqDocPair>> preGpuAlgoBatch(const vector<ItemCpu> &reqs,
                                           const vector<ItemCpu> &docs,
                                           int numToRetrieve,
                                           float minScore,
                                           float maxScore);

#endif // PRE_GPU_CUH