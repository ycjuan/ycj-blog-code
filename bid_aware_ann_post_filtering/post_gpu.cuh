#ifndef POST_GPU_CUH
#define POST_GPU_CUH

#include "data_struct.cuh"
#include "common.cuh"
#include "topk.cuh"
#include "misc.cuh"
#include "util.cuh"
#include "config.cuh"

#include <random>
#include <iostream>
#include <algorithm>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

using namespace std;

//TODO: rename k to numToRetrieve
vector<vector<ReqDocPair>> postGpuAlgoBatch(const vector<CentroidCpu> &centroids,
                                            const vector<ItemCpu> &reqs,
                                            const vector<ItemCpu> &docs,
                                            int k);

#endif // POST_GPU_CUH