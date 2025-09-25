#include "data.cuh"
#include "util.cuh"
#include <random>

namespace BatchScalability
{

__global__ void kernelGpuNaive1(Data data)
{
    uint64_t threadId = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int reqIdx = threadId / data.numDocs;
    int docIdx = threadId % data.numDocs;
    if (reqIdx < data.numReqs && docIdx < data.numDocs)
    {
        double rst = 0;
        for (int embIdx = 0; embIdx < data.embDim; embIdx++)
        {
            float reqVal = data.d_reqData[getMemAddrReq(reqIdx, embIdx, data.numReqs, data.embDim)];
            float docVal = data.d_docData[getMemAddrDoc(docIdx, embIdx, data.numDocs, data.embDim)];
            rst += std::sqrt(reqVal * docVal);
        }
        data.d_rstDataGpu[getMemAddrRst(reqIdx, docIdx, data.numReqs, data.numDocs)] = rst;    
    }
}

void methodGpuNaive1(Data& data)
{
    uint64_t blockSize = kBlockSize;
    uint64_t gridSize = (data.numReqs * data.numDocs + blockSize - 1) / blockSize;
    kernelGpuNaive1<<<gridSize, blockSize>>>(data);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
}

__global__ void kernelGpuNaive2(Data data)
{
    int threadId = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int reqIdx = threadId % data.numReqs;
    int docIdx = threadId / data.numReqs;
    double rst = 0;
    for (int embIdx = 0; embIdx < data.embDim; embIdx++)
    {
        float reqVal = data.d_reqData[getMemAddrReq(reqIdx, embIdx, data.numReqs, data.embDim)];
        float docVal = data.d_docData[getMemAddrDoc(docIdx, embIdx, data.numDocs, data.embDim)];
        rst += std::sqrt(reqVal * docVal);
    }
    data.d_rstDataGpu[getMemAddrRst(reqIdx, docIdx, data.numReqs, data.numDocs)] = rst;
}

void methodGpuNaive2(Data& data)
{
    uint64_t blockSize = kBlockSize;
    uint64_t gridSize = (data.numReqs * data.numDocs + blockSize - 1) / blockSize;
    kernelGpuNaive2<<<gridSize, blockSize>>>(data);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
}

__global__ void kernelGpuNaive3(Data data)
{
    int reqIdx = blockIdx.y;
    int docIdx = blockIdx.x;
    double rst = 0;
    for (int embIdx = 0; embIdx < data.embDim; embIdx++)
    {
        float reqVal = data.d_reqData[getMemAddrReq(reqIdx, embIdx, data.numReqs, data.embDim)];
        float docVal = data.d_docData[getMemAddrDoc(docIdx, embIdx, data.numDocs, data.embDim)];
        rst += std::sqrt(reqVal * docVal);
    }
    data.d_rstDataGpu[getMemAddrRst(reqIdx, docIdx, data.numReqs, data.numDocs)] = rst;
}

void methodGpuNaive3(Data& data)
{
    dim3 blockSize(1024 / data.numReqs, data.numReqs);
    dim3 gridSize((data.numDocs + blockSize.x - 1) / blockSize.x, 1);
    kernelGpuNaive3<<<gridSize, blockSize>>>(data);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
}

} // namespace BatchScalability