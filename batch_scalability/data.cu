#include <cuda_runtime.h>
#include "data.cuh"

#include "util.cuh"

namespace BatchScalability {

Data genData(int numReqs, int numDocs, int embDim)
{
    Data data;
    data.numReqs = numReqs;
    data.numDocs = numDocs;
    data.embDim = embDim;
    CHECK_CUDA(cudaMallocManaged(&data.d_docData, numDocs * embDim * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&data.d_reqData, numReqs * embDim * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&data.h_rstDataCpu, numDocs * numReqs * sizeof(float) ));
    CHECK_CUDA(cudaMallocManaged(&data.d_rstDataGpu, numDocs * numReqs * sizeof(float) ));
    return data;
}

void freeData(Data& data)
{
    if (data.d_docData != nullptr) {
        cudaFree(data.d_docData);
    }
    if (data.d_reqData != nullptr) {
        cudaFree(data.d_reqData);
    }
    if (data.h_rstDataCpu != nullptr) {
        cudaFreeHost(data.h_rstDataCpu);
    }
    if (data.d_rstDataGpu != nullptr) {
        cudaFree(data.d_rstDataGpu);
    }
}

} // namespace BatchScalability