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

int kNumDocs = 1 << 18;
int kNumReqs = 1 << 7;
int kNumT1 = 1 << 5;
int kNumTrials = 100;
MemLayout kMemLayoutDoc = ROW_MAJOR; 
// IMPORTANT: Don't change this. a_frag in WMMA requires ROW_MAJOR
MemLayout kMemLayoutReq = ROW_MAJOR; 
// IMPORTANT: Don't change this. b_frag in WMMA requires COL_MAJOR. 
// However, since the matrix here has a shape of (numReqs, numInt), setting ROW_MAJOR here is equivalent to COL_MAJOR of a (numInt, numReqs) matrix
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
    data.numInt = kNumT1;
    data.docMemLayout = kMemLayoutDoc;
    data.reqMemLayout = kMemLayoutReq;
    data.rstLayoutCpu = kMemLayoutRstCpu;
    data.rstLayoutGpuKernel = kMemLayoutRstGpuKernel;
    data.rstLayoutGpuCublas = kMemLayoutRstGpuTensor;
    data.print();
    
    CHECK_CUDA(cudaMallocManaged(&data.d_doc, (size_t)data.numDocs * data.numInt * sizeof(T_QUANT)));
    CHECK_CUDA(cudaMallocManaged(&data.d_req, (size_t)data.numReqs * data.numInt * sizeof(T_QUANT)));
    CHECK_CUDA(cudaMallocManaged(&data.d_rst_kernel, (size_t)data.numDocs * data.numReqs * sizeof(T_RST)));
    CHECK_CUDA(cudaMallocManaged(&data.d_rst_wmma, (size_t)data.numDocs * data.numReqs * sizeof(T_RST)));
    CHECK_CUDA(cudaMallocHost(&data.h_rst_cpu, (size_t)data.numDocs * data.numReqs * sizeof(T_RST)));


    default_random_engine generator;
    uniform_int_distribution<T_QUANT> distribution;

    T_QUANT uid = 0;
    for (int i = 0; i < data.numDocs; i++)
        for (int k = 0; k < data.numInt; k++)
            data.d_doc[getMemAddr(i, k, data.numDocs, data.numInt, data.docMemLayout)] = uid++;
    uid = 0;

    for (int j = 0; j < data.numReqs; j++)
    {
        for (int k = 0; k < data.numInt; k++)
        {
            size_t addr = getMemAddr(j, k, data.numReqs, data.numInt, data.reqMemLayout);
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
            T_RST cpuVal = data.h_rst_cpu[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutCpu)];
            T_RST gpuKernelVal = data.d_rst_kernel[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutGpuKernel)];
            T_RST gpuWmma = data.d_rst_wmma[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutGpuCublas)];

            if (false)
            {
                cout << "Kernel error at (" << i << ", " << j << "): " << cpuVal << " != " << gpuKernelVal << endl;
                return;
            }
            
            if (cpuVal != gpuWmma)
            {
                if (numPrinted++ < 256)
                    cout << "Wmma error at (" << i << ", " << j << "): " << cpuVal << " != " << gpuWmma << endl;
            }
        }
    }
}

int main()
{
    Data data = genData();
    Setting setting;
    setting.kNumTrials = kNumTrials;

    quantGpuCuda(data, setting);
    quantCpu(data, setting);
    quantWmmaUnroll(data, setting);

    checkData(data);

    data.free();

    return 0;
}
