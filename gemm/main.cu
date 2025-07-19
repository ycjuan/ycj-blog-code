#include <string>
#include <stdexcept>
#include <iostream>
#include <random>
#include <sstream>
#include <cublas_v2.h>
#include <type_traits>

#include "util.cuh"
#include "data.cuh"
#include "method_dp_cpu.cuh"
#include "method_dp_gpu_cublas.cuh"
#include "method_dp_gpu_naive.cuh"
#include "method_mlp_cpu.cuh"

using namespace std;

typedef half T; 
// IMPORTANT!!! only __nv_bfloat16 and half are supported for now

int kNumDocs = 1 << 20;
int kNumReqs = 1 << 0;
int kEmbDim = 1 << 10;
int kNumTrials = 100;

MemLayout kMemLayoutDoc = COL_MAJOR;
MemLayout kMemLayoutReq = ROW_MAJOR;
MemLayout kMemLayoutRstCpu = COL_MAJOR;
MemLayout kMemLayoutRstGpuCuda = COL_MAJOR;
MemLayout kMemLayoutRstGpuCublas = COL_MAJOR;

template <typename T>
Data<T> genData()
{
    Data<T> data;
    data.numDocs = kNumDocs;
    data.numReqs = kNumReqs;
    data.embDim = kEmbDim;
    data.docMemLayout = kMemLayoutDoc;
    data.reqMemLayout = kMemLayoutReq;
    data.rstLayoutCpu = kMemLayoutRstCpu;
    data.rstLayoutGpuKernel = kMemLayoutRstGpuCuda;
    data.rstLayoutGpuCublas = kMemLayoutRstGpuCublas;
    data.print();
    
    CHECK_CUDA(cudaMallocManaged(&data.d_doc, (size_t)data.numDocs * data.embDim * sizeof(T)));
    CHECK_CUDA(cudaMallocManaged(&data.d_req, (size_t)data.numReqs * data.embDim * sizeof(T)));
    CHECK_CUDA(cudaMallocManaged(&data.d_wa, (size_t)data.embDim * data.hiddenDim * sizeof(T)));
    CHECK_CUDA(cudaMallocManaged(&data.d_wb, (size_t)data.hiddenDim * sizeof(T)));
    CHECK_CUDA(cudaMallocManaged(&data.d_rst_cublas, (size_t)data.numDocs * data.numReqs * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&data.d_rst_mlp_gpu_naive, (size_t)data.numDocs * data.numReqs * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&data.d_rst_dp_gpu_naive, (size_t)data.numDocs * data.numReqs * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&data.h_rst_cpu, (size_t)data.numDocs * data.numReqs * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&data.h_rst_mlp_cpu, (size_t)data.numDocs * data.numReqs * sizeof(float)));

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

template <typename T>
void checkData(Data<T> data)
{
    for (int i = 0; i < data.numDocs; i++)
    {
        for (int j = 0; j < data.numReqs; j++)
        {
            float cpuVal = data.h_rst_cpu[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutCpu)];
            float gpuCublasVal = data.d_rst_cublas[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutGpuCublas)];
            float gpuNaiveVal = data.d_rst_dp_gpu_naive[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutGpuKernel)];

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
    }
}

int main()
{
    Data<T> data = genData<T>();

    methodDpCublas(data, kNumTrials);

    methodDpCpu(data);

    methodDpGpuNaive(data, kNumTrials);

    methodMlpCpu(data);

    checkData(data);

    data.free();

    return 0;
}