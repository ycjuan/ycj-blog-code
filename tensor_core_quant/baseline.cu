#include <bitset>

#include "common.cuh"
#include "util.cuh"

void quantCpu(Data data, Setting setting)
{
    Timer timer;
    timer.tic();
    #pragma omp parallel for
    for (int i = 0; i < data.numDocs; i++)
    {
        for (int j = 0; j < data.numReqs; j++)
        {
            T2 totalCount = 0;
            for (int k = 0; k < data.numT1; k++)
            {
                T1 reqVal = data.d_req[getMemAddr(j, k, data.numReqs, data.numT1, data.reqMemLayout)];
                T1 docVal = data.d_doc[getMemAddr(i, k, data.numDocs, data.numT1, data.docMemLayout)];
                T1 bitwiseRst = ~ (reqVal ^ docVal);
                uint64_t bitwiseRst64 = uint64_t(bitwiseRst);
                bitset<64> bits(bitwiseRst64);
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
        T2 totalCount = 0;
        for (int k = 0; k < data.numT1; k++)
        {
            T1 reqVal = data.d_req[getMemAddr(j, k, data.numReqs, data.numT1, data.reqMemLayout)];
            T1 docVal = data.d_doc[getMemAddr(i, k, data.numDocs, data.numT1, data.docMemLayout)];    
            T1 bitwiseRst = ~ (reqVal ^ docVal);
            uint64_t bitwiseRst64 = uint64_t(bitwiseRst);
            totalCount += __popcll(bitwiseRst64); // This counts the number of "1" in the 64bit bitwiseAnd
        }
        data.d_rst_kernel[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutGpuKernel)] = totalCount;
    }
}

void quantKernel(Data data, Setting setting)
{
    int blockSize = 512;
    int gridSize = (size_t(data.numDocs) * data.numReqs + blockSize - 1) / blockSize;
    CudaTimer timer;
    for (int t = -3; t < setting.kNumTrials; t++)
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
    cout << "Kernel time: " << timer.tocMs() / setting.kNumTrials << " ms" << endl;
}
