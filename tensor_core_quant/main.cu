#include <string>
#include <stdexcept>
#include <iostream>
#include <random>
#include <sstream>
#include <cublas_v2.h>
#include <type_traits>

#include "util.cuh"
#include "common.cuh"
#include "methods.cuh"

using namespace std;

int kNumDocs = 1 << 4;
int kNumReqs = 1 << 3;
int kNumT1 = 1 << 2;
int kNumTrials = 100;
MemLayout kMemLayoutDoc = ROW_MAJOR;
MemLayout kMemLayoutReq = ROW_MAJOR;
MemLayout kMemLayoutRstCpu = COL_MAJOR;
MemLayout kMemLayoutRstGpuKernel = COL_MAJOR;
MemLayout kMemLayoutRstGpuTensor = ROW_MAJOR;

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
    data.numT1 = kNumT1;
    data.docMemLayout = kMemLayoutDoc;
    data.reqMemLayout = kMemLayoutReq;
    data.rstLayoutCpu = kMemLayoutRstCpu;
    data.rstLayoutGpuKernel = kMemLayoutRstGpuKernel;
    data.rstLayoutGpuCublas = kMemLayoutRstGpuTensor;
    data.print();
    
    CHECK_CUDA(cudaMallocManaged(&data.d_doc, (size_t)data.numDocs * data.numT1 * sizeof(T1)));
    CHECK_CUDA(cudaMallocManaged(&data.d_req, (size_t)data.numReqs * data.numT1 * sizeof(T1)));
    CHECK_CUDA(cudaMallocManaged(&data.d_rst_kernel, (size_t)data.numDocs * data.numReqs * sizeof(T2)));
    CHECK_CUDA(cudaMallocManaged(&data.d_rst_wmma, (size_t)data.numDocs * data.numReqs * sizeof(T2)));
    CHECK_CUDA(cudaMallocHost(&data.h_rst_cpu, (size_t)data.numDocs * data.numReqs * sizeof(T2)));


    default_random_engine generator;
    uniform_int_distribution<T1> distribution;

    T1 uid = 0;
    for (int i = 0; i < data.numDocs; i++)
        for (int k = 0; k < data.numT1; k++)
            data.d_doc[getMemAddr(i, k, data.numDocs, data.numT1, data.docMemLayout)] = uid++;
    uid = 0;

    for (int j = 0; j < data.numReqs; j++)
    {
        for (int k = 0; k < data.numT1; k++)
        {
            size_t addr = getMemAddr(j, k, data.numReqs, data.numT1, data.reqMemLayout);
            data.d_req[addr] = uid++;
        }
    }
    return data;
}

void checkData(Data data)
{
    int numPrinted = 0;
    for (int i = 0; i < data.numDocs; i++)
    {
        for (int j = 0; j < data.numReqs; j++)
        {
            T2 cpuVal = data.h_rst_cpu[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutCpu)];
            T2 gpuKernelVal = data.d_rst_kernel[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutGpuKernel)];
            T2 gpuWmma = data.d_rst_wmma[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutGpuCublas)];

            if (false)
            {
                cout << "Kernel error at (" << i << ", " << j << "): " << cpuVal << " != " << gpuKernelVal << endl;
                return;
            }
            
            if (cpuVal != gpuWmma)
            {
                if (numPrinted++ < 256)
                    cout << "Wmma error at (" << i << ", " << j << "): " << cpuVal << " != " << gpuWmma << endl;
                //return;
            }
        }
    }
}

int main()
{
    Data data = genData();
    Setting setting;
    setting.kNumTrials = kNumTrials;

    //quantKernel(data, setting);
    quantCpu(data, setting);
    quantWMMA(data, setting);

    checkData(data);

    data.free();

    return 0;
}
