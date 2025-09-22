#include "data.cuh"
#include "util.cuh"

namespace BatchScalability
{

__global__ void kernelGpuNaive(Data data)
{
    int reqIdx = blockIdx.x;
    int docIdx = blockIdx.y;
    float rst = 0;
    for (int embIdx = 0; embIdx < data.embDim; embIdx++)
    {
        float reqVal = data.d_reqData[getMemAddrReq(reqIdx, embIdx, data.numReqs, data.embDim)];
        float docVal = data.d_docData[getMemAddrDoc(docIdx, embIdx, data.numDocs, data.embDim)];
        rst += std::sqrt(reqVal * reqVal);
    }
    data.d_rstDataGpu[getMemAddrRst(reqIdx, docIdx, data.numReqs, data.numDocs)] = rst;
}

void methodGpuNaive(Data& data)
{
    dim3 blockSize(data.numReqs, 1024 / data.numReqs);
    dim3 gridSize(1, (data.numDocs + blockSize.y - 1) / blockSize.y);
    kernelGpuNaive<<<gridSize, blockSize>>>(data);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
}

} // namespace BatchScalability