#include <vector>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <cassert>
#include <numeric>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <string>

#include "topk.cuh"
#include "common.cuh"
#include "util.cuh"

using namespace std;

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "[topk_with_bucket_sort.cu] CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

void TopkSampling::init()
{

}

void TopkSampling::reset()
{

}

void TopkSampling::retrieveTopk(TopkParam &param)
{
    CudaTimer timerTotal;
    CudaTimer timerApprox;
    timerTotal.tic();
    timerApprox.tic();

    // Step1 - Sample
    sample(param);

    // Step2 - Sort
    float threshold = 0;
    findThreshold(param, threshold);

    // Step3 - Copy eligible 
    size_t numCopied = 0;
    copyEligible(param, threshold, numCopied);
    param.gpuApproxTimeMs = timerApprox.tocMs();

    // Step4 - retreiveExact
    retrieveExact(param);

    param.gpuTotalTimeMs = timerTotal.tocMs();
}
