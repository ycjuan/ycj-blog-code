#include <string>
#include <stdexcept>
#include <iostream>
#include <random>

#include "timer.cuh"
#include "data.cuh"
#include "method_dp_cpu.cuh"
#include "method_dp_gpu_cublas.cuh"
#include "method_dp_gpu_naive.cuh"
#include "method_mlp_cpu.cuh"
#include "method_mlp_gpu.cuh"

using namespace std;
size_t kNumDocs = 1000;
size_t kNumReqs = 32;
size_t kEmbDim = 64;
size_t kHiddenDim = 32;
size_t kNumTrials = 100;


#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

Data genData()
{
    Data data;
    data.numDocs = kNumDocs;
    data.numReqs = kNumReqs;
    data.embDim = kEmbDim;
    data.hiddenDim = kHiddenDim;
    data.print();
    
    CHECK_CUDA(cudaMallocManaged(&data.d_doc, data.numDocs * data.embDim * sizeof(T)));
    CHECK_CUDA(cudaMallocManaged(&data.d_req, data.numReqs * data.embDim * sizeof(T)));
    CHECK_CUDA(cudaMallocManaged(&data.d_wa, data.embDim * data.hiddenDim * sizeof(T)));
    CHECK_CUDA(cudaMallocManaged(&data.d_wb, data.hiddenDim * sizeof(T)));
    CHECK_CUDA(cudaMallocManaged(&data.d_rst_dp_gpu_naive, data.numDocs * data.numReqs * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&data.d_rst_dp_gpu_cublas, data.numDocs * data.numReqs * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&data.d_rst_mlp_gpu, data.numDocs * data.numReqs * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&data.h_rst_dp_cpu, data.numDocs * data.numReqs * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&data.h_rst_mlp_cpu, data.numDocs * data.numReqs * sizeof(float)));

    default_random_engine generator;
    uniform_real_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < data.numDocs * data.embDim; i++)
        data.d_doc[i] = (T)distribution(generator);
    for (int i = 0; i < data.numReqs * data.embDim; i++)
        data.d_req[i] = (T)distribution(generator);
    for (int i = 0; i < data.embDim * data.hiddenDim; i++)
        data.d_wa[i] = (T)distribution(generator);
    for (int i = 0; i < data.hiddenDim; i++)
        data.d_wb[i] = (T)distribution(generator);

    return data;
}

void checkData(Data data)
{
    for (int i = 0; i < data.numDocs; i++)
    {
        for (int j = 0; j < data.numReqs; j++)
        {
            // Check DP results
            {
                float cpuVal = data.h_rst_dp_cpu[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstDpCpuMemLayout)];
                float gpuCublasVal = data.d_rst_dp_gpu_cublas[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstDpGpuCublasMemLayout)];
                float gpuNaiveVal = data.d_rst_dp_gpu_naive[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstDpGpuNaiveMemLayout)];

                if (abs(cpuVal - gpuCublasVal) / abs(cpuVal) > 1e-3)
                {
                    cout << "Cublas error at (" << i << ", " << j << "): " << cpuVal << " != " << gpuCublasVal << endl;
                    return;
                }

                if (abs(cpuVal - gpuNaiveVal) / abs(cpuVal) > 1e-3)
                {
                    cout << "Naive GPU error at (" << i << ", " << j << "): " << cpuVal << " != " << gpuNaiveVal << endl;
                    return;
                }
            }

            // Check MLP results
            {
                float cpuVal = data.h_rst_mlp_cpu[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstMlpCpuMemLayout)];
                float gpuVal = data.d_rst_mlp_gpu[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstMlpGpuMemLayout)];

                // Enable the check below after GPU MLP implementation is complete
                bool gpuMlpImplemented = false;
                if (!gpuMlpImplemented)
                {
                    continue;
                }
                if (abs(cpuVal - gpuVal) / abs(cpuVal) > 1e-3)
                {
                    cout << "MLP error at (" << i << ", " << j << "): " << cpuVal << " != " << gpuVal << endl;
                    return;
                }
            }
        }
    }
}

int main()
{
    Data data = genData();

    methodDpCpu(data);

    methodDpGpuNaive(data, kNumTrials);

    methodDpCublas(data, kNumTrials);

    methodMlpCpu(data);

    methodMlpGpu(data, kNumTrials);

    checkData(data);

    data.free();

    return 0;
}