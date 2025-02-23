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
            T_RST totalCount = 0;
            for (int k = 0; k < data.numInt; k++)
            {
                T_QUANT reqVal = data.d_req[getMemAddr(j, k, data.numReqs, data.numInt, data.reqMemLayout)];
                T_QUANT docVal = data.d_doc[getMemAddr(i, k, data.numDocs, data.numInt, data.docMemLayout)];
                T_QUANT bitwiseRst = ~(reqVal ^ docVal);
                bitset<32> bits(bitwiseRst);
                int count = bits.count();
                totalCount += count;
            }
            data.h_rst_cpu[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutCpu)] = totalCount;
        }
    }
    cout << "CPU time: " << timer.tocMs() << " ms" << endl;
}

__global__ void quantKernel(Data data)
{
    size_t threadId = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int i = threadId / data.numReqs;
    int j = threadId % data.numReqs;

    if (i < data.numDocs && j < data.numReqs)
    {
        T_RST totalCount = 0;
        for (int k = 0; k < data.numInt; k++)
        {
            T_QUANT reqVal = data.d_req[getMemAddr(j, k, data.numReqs, data.numInt, data.reqMemLayout)];
            T_QUANT docVal = data.d_doc[getMemAddr(i, k, data.numDocs, data.numInt, data.docMemLayout)];    
            T_QUANT bitwiseRst = ~ (reqVal ^ docVal);
            uint64_t bitwiseRst64 = uint64_t(bitwiseRst);
            totalCount += __popcll(bitwiseRst64); // This counts the number of "1" in the 64bit bitwiseAnd
        }
        data.d_rst_kernel[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutGpuCuda)] = totalCount;
    }
}

void quantGpuCuda(Data data, Setting setting)
{
    int blockSize = 512;
    size_t gridSize = (size_t(data.numDocs) * data.numReqs + blockSize - 1) / blockSize;
    cout << "gridSize (cuda): " << gridSize << endl;
    CudaTimer timer;
    for (int t = -3; t < setting.kNumTrials; t++)
    {
        if (t == 0)
            timer.tic();
        quantKernel<<<gridSize, blockSize>>>(data);
        cudaDeviceSynchronize();
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess)
        {
            ostringstream oss;
            oss << "Kernel launch failed with error: " << cudaGetErrorString(status) << "\n";
            throw runtime_error(oss.str());
        }
    }
    cout << "Cuda time: " << timer.tocMs() / setting.kNumTrials << " ms" << endl;
}
