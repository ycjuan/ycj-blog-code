#include "data.cuh"
#include "util.cuh"

namespace BatchScalability
{

__global__ void kernelGpu1(Data data)
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

void methodGpu1(Data& data)
{
    uint64_t blockSize = kBlockSize;
    uint64_t gridSize = (data.numReqs * data.numDocs + blockSize - 1) / blockSize;
    kernelGpu1<<<gridSize, blockSize>>>(data);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
}

__global__ void kernelGpu2(Data data)
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

void methodGpu2(Data& data)
{
    uint64_t blockSize = kBlockSize;
    uint64_t gridSize = (data.numReqs * data.numDocs + blockSize - 1) / blockSize;
    kernelGpu2<<<gridSize, blockSize>>>(data);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
}

__global__ void kernelGpu3(Data data)
{
    int docIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int reqIdx = threadIdx.y;
    double rst = 0;
    for (int embIdx = 0; embIdx < data.embDim; embIdx++)
    {
        float reqVal = data.d_reqData[getMemAddrReq(reqIdx, embIdx, data.numReqs, data.embDim)];
        float docVal = data.d_docData[getMemAddrDoc(docIdx, embIdx, data.numDocs, data.embDim)];
        rst += std::sqrt(reqVal * docVal);
    }
    data.d_rstDataGpu[getMemAddrRst(reqIdx, docIdx, data.numReqs, data.numDocs)] = rst;
}

void methodGpu3(Data& data)
{
    dim3 blockSize(1024 / data.numReqs, data.numReqs);
    dim3 gridSize((data.numDocs + blockSize.x - 1) / blockSize.x, 1);
    kernelGpu3<<<gridSize, blockSize>>>(data);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
}

constexpr int kWarpSize = 32;
constexpr int kLaneMask = kWarpSize - 1;

__global__ void kernelGpu4(Data data)
{
    int numThreads = gridDim.x * blockDim.x;
    int numWarps = numThreads / kWarpSize;
    int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpIdx = globalThreadIdx / kWarpSize;
    int laneIdx = threadIdx.x & kLaneMask;

    for (int docIdx = warpIdx; docIdx < data.numDocs; docIdx += numWarps)
    {
        for (int reqIdx = 0; reqIdx < data.numReqs; reqIdx++)
        {
            float rst = 0;
            for (int embIdx = laneIdx; embIdx < data.embDim; embIdx += kWarpSize)
            {
                float reqVal = data.d_reqData[getMemAddrReq(reqIdx, embIdx, data.numReqs, data.embDim)];
                float docVal = data.d_docData[getMemAddrDoc(docIdx, embIdx, data.numDocs, data.embDim)];
                rst += std::sqrt(reqVal * docVal);
            }
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                rst += __shfl_down_sync(0xFFFFFFFF, rst, offset);
            }
            if (laneIdx == 0)
            {
                data.d_rstDataGpu[getMemAddrRst(reqIdx, docIdx, data.numReqs, data.numDocs)] = rst;
            }
        }
    }
}

void methodGpu4(Data& data)
{
    uint64_t blockSize = 1024;
    uint64_t gridSize = 116;
    kernelGpu4<<<gridSize, blockSize>>>(data);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
}

__global__ void kernelGpu5(Data data)
{
    int numThreads = gridDim.x * blockDim.x;
    int numWarps = numThreads / kWarpSize;
    int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpIdx = globalThreadIdx / kWarpSize;
    int laneIdx = threadIdx.x & kLaneMask;
    int reqIdx = threadIdx.y;

    for (int docIdx = warpIdx; docIdx < data.numDocs; docIdx += numWarps)
    {

        float rst = 0;
        for (int embIdx = laneIdx; embIdx < data.embDim; embIdx += kWarpSize)
        {
            float reqVal = data.d_reqData[getMemAddrReq(reqIdx, embIdx, data.numReqs, data.embDim)];
            float docVal = data.d_docData[getMemAddrDoc(docIdx, embIdx, data.numDocs, data.embDim)];
            rst += std::sqrt(reqVal * docVal);
        }
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            rst += __shfl_down_sync(0xFFFFFFFF, rst, offset);
        }
        if (laneIdx == 0)
        {
            data.d_rstDataGpu[getMemAddrRst(reqIdx, docIdx, data.numReqs, data.numDocs)] = rst;
        }
    }
}

void methodGpu5(Data& data)
{
    dim3 blockSize(1024 / data.numReqs, data.numReqs);
    dim3 gridSize((data.numDocs + blockSize.x - 1) / blockSize.x, 1);
    kernelGpu5<<<gridSize, blockSize>>>(data);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
}

} // namespace BatchScalability