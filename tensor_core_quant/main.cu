#include <string>
#include <stdexcept>
#include <iostream>
#include <random>
#include <sstream>
#include <cublas_v2.h>
#include <type_traits>

#include "util.cuh"
#include "common.cuh"
#include "baseline.cuh"

using namespace std;

int kNumDocs = 1 << 20;
int kNumReqs = 1 << 4;
int kNumInt64 = 1 << 3;
int kNumTrials = 100;
MemLayout kMemLayoutDoc = COL_MAJOR;
MemLayout kMemLayoutReq = ROW_MAJOR;
MemLayout kMemLayoutRstCpu = COL_MAJOR;
MemLayout kMemLayoutRstGpuKernel = COL_MAJOR;
MemLayout kMemLayoutRstGpuTensor = COL_MAJOR;

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
    data.numInt64 = kNumInt64;
    data.docMemLayout = kMemLayoutDoc;
    data.reqMemLayout = kMemLayoutReq;
    data.rstLayoutCpu = kMemLayoutRstCpu;
    data.rstLayoutGpuKernel = kMemLayoutRstGpuKernel;
    data.rstLayoutGpuCublas = kMemLayoutRstGpuTensor;
    data.print();
    
    CHECK_CUDA(cudaMallocManaged(&data.d_doc, (size_t)data.numDocs * data.numInt64 * sizeof(uint64_t)));
    CHECK_CUDA(cudaMallocManaged(&data.d_req, (size_t)data.numReqs * data.numInt64 * sizeof(uint64_t)));
    CHECK_CUDA(cudaMallocManaged(&data.d_rst_kernel, (size_t)data.numDocs * data.numReqs * sizeof(uint16_t)));
    CHECK_CUDA(cudaMallocManaged(&data.d_rst_cublas, (size_t)data.numDocs * data.numReqs * sizeof(uint16_t)));
    CHECK_CUDA(cudaMallocHost(&data.h_rst_cpu, (size_t)data.numDocs * data.numReqs * sizeof(uint16_t)));


    default_random_engine generator;
    uniform_int_distribution<uint64_t> distribution;
    for (int i = 0; i < data.numDocs * data.numInt64; i++)
        data.d_doc[i] = distribution(generator);
    for (int i = 0; i < data.numReqs * data.numInt64; i++)
        data.d_req[i] = distribution(generator);

    return data;
}

void checkData(Data data)
{
    for (int i = 0; i < data.numDocs; i++)
    {
        for (int j = 0; j < data.numReqs; j++)
        {
            uint16_t cpuVal = data.h_rst_cpu[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutCpu)];
            uint16_t gpuKernelVal = data.d_rst_kernel[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutGpuKernel)];
            uint16_t gpuCublasVal = data.d_rst_cublas[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutGpuCublas)];

            if (abs(cpuVal - gpuKernelVal) / abs(gpuKernelVal) > 1e-3)
            {
                cout << "Kernel error at (" << i << ", " << j << "): " << cpuVal << " != " << gpuKernelVal << endl;
                return;
            }
            
            /*
            if (abs(cpuVal - gpuCublasVal) / abs(gpuKernelVal) > 1e-3)
            {
                cout << "Cublas error at (" << i << ", " << j << "): " << cpuVal << " != " << gpuCublasVal << endl;
                return;
            }
            */
        }
    }
}

int main()
{
    Data data = genData();
    Setting setting;
    setting.kNumTrials = kNumTrials;

    matMulKernel(data, setting);
    matMulCpu(data, setting);

    checkData(data);

    data.free();

    return 0;
}
