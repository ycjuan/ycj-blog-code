#include <string>
#include <stdexcept>
#include <iostream>
#include <random>
#include <sstream>
#include <cublas_v2.h>
#include <type_traits>
#include <bitset>

#include "util.cuh"

using namespace std;

enum MemLayout
{
    ROW_MAJOR,
    COL_MAJOR
};

int kNumDocs = 1 << 20;
int kNumReqs = 1 << 0;
int kNumInt64 = 1 << 10;
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

__device__ __host__ size_t getMemAddr(int i, int j, int M, int N, MemLayout layout)
{
    if (layout == ROW_MAJOR)
        return (size_t)i * N + j;
    else
        return (size_t)j * M + i;
}

struct Data
{
    int numDocs;
    int numReqs;
    int numInt64;
    uint64_t *d_doc; // M=numDocs x N=numInt64
    uint64_t *d_req; // M=numReqs x N=numInt64
    uint16_t *d_rst_kernel; // M=numDocs x N=numReqs
    uint16_t *d_rst_cublas; // M=numDocs x N=numReqs
    uint16_t *h_rst_cpu;
    MemLayout docMemLayout;
    MemLayout reqMemLayout;
    MemLayout rstLayoutCpu;
    MemLayout rstLayoutGpuKernel;
    MemLayout rstLayoutGpuCublas;

    void free()
    {
        cudaFree(d_doc);
        cudaFree(d_req);
        cudaFree(d_rst_kernel);
        cudaFree(d_rst_cublas);
        cudaFreeHost(h_rst_cpu);
    }

    void print()
    {
        ostringstream oss;
        oss << "numDocs: " << numDocs << ", numReqs: " << numReqs << ", numInt64: " << numInt64 << endl;
        oss << "docMemLayout: " << (docMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "reqMemLayout: " << (reqMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstLayoutCpu: " << (rstLayoutCpu == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstLayoutGpuKernel: " << (rstLayoutGpuKernel == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstLayoutGpuCublas: " << (rstLayoutGpuCublas == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        cout << oss.str();
    }
};

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
            
            if (abs(cpuVal - gpuCublasVal) / abs(gpuKernelVal) > 1e-3)
            {
                cout << "Cublas error at (" << i << ", " << j << "): " << cpuVal << " != " << gpuCublasVal << endl;
                return;
            }
        }
    }
}

void matMulCpu(Data data)
{
    Timer timer;
    timer.tic();
    #pragma omp parallel for
    for (int i = 0; i < data.numDocs; i++)
    {
        for (int j = 0; j < data.numReqs; j++)
        {
            uint16_t totalCount = 0;
            for (int k = 0; k < data.numInt64; k++)
            {
                uint64_t reqVal = data.d_req[getMemAddr(j, k, data.numReqs, data.numInt64, data.reqMemLayout)];
                uint64_t docVal = data.d_doc[getMemAddr(i, k, data.numDocs, data.numInt64, data.docMemLayout)];
                uint64_t bitwiseRst = ~ (reqVal ^ docVal);
                bitset<64> bits(bitwiseRst);
                totalCount += bits.count();
            }
            data.h_rst_cpu[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutCpu)] = totalCount;
        }
    }
    cout << "CPU time: " << timer.tocMs() << " ms" << endl;
}

__global__ void matMul(Data data)
{
    int threadId = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int i = threadId / data.numReqs;
    int j = threadId % data.numReqs;

    if (i < data.numDocs && j < data.numReqs)
    {
        uint16_t totalCount = 0;
        for (int k = 0; k < data.numInt64; k++)
        {
            uint64_t reqVal = data.d_req[getMemAddr(j, k, data.numReqs, data.numInt64, data.reqMemLayout)];
            uint64_t docVal = data.d_doc[getMemAddr(i, k, data.numDocs, data.numInt64, data.docMemLayout)];            
            uint64_t bitwiseRst = ~ (reqVal ^ docVal);
            totalCount += __popcll(bitwiseRst); // This counts the number of "1" in the 64bit bitwiseAnd
        }
        data.d_rst_kernel[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutGpuKernel)] = totalCount;
    }
}

void matMulKernel(Data data)
{
    int blockSize = 512;
    int gridSize = size_t(data.numDocs) * data.numReqs / blockSize;
    CudaTimer timer;
    for (int t = -3; t < kNumTrials; t++)
    {
        if (t == 0)
            timer.tic();
        matMul<<<gridSize, blockSize>>>(data);
        cudaDeviceSynchronize();
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess)
        {
            ostringstream oss;
            oss << "Kernel launch failed with error: " << cudaGetErrorString(status) << "\n";
            throw runtime_error(oss.str());
        }
    }
    cout << "Kernel time: " << timer.tocMs() / kNumTrials << " ms" << endl;
}

int main()
{
    Data data = genData();

    matMulKernel(data);
    matMulCpu(data);

    checkData(data);

    data.free();

    return 0;
}
