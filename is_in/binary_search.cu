#include "methods.cuh"
#include "util.cuh"
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

using namespace std;

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

struct CopyPredicator
{
    __device__ bool operator()(const Doc &doc)
    {
        return doc.isIn == 1;
    }
};

void __global__ methodGpuBinarySearchKernel(Doc *d_doc, uint64_t *d_list, int numDocs, int listSize)
{
    int docIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (docIdx >= numDocs)
        return;

    Doc &doc = d_doc[docIdx];
    doc.isIn = false;

    int low = 0;
    int high = listSize - 1;

    while (low <= high) 
    {
        int mid = low + (high - low) / 2;

        if (d_list[mid] == doc.docHash) 
        {
            doc.isIn = true;
            break;
        }

        if (d_list[mid] < doc.docHash)
            low = mid + 1;
        else
            high = mid - 1;
    }
}

void methodGpuBinarySearch(Data &data, Setting setting)
{
    int blockSize = 256;
    int numBlocks = (data.numDocs + blockSize - 1) / blockSize;

    CudaTimer timer;
    for (int t = -3; t < setting.numTrials; t++)
    {
        CHECK_CUDA(cudaMemcpy(data.d_docGpu, data.doc1D.data(), data.numDocs * sizeof(Doc), cudaMemcpyHostToDevice));

        timer.tic();
        methodGpuBinarySearchKernel<<<numBlocks, blockSize>>>(data.d_docGpu, data.d_list, data.numDocs, data.listSize);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());

        Doc *endPtr = thrust::copy_if(thrust::device, data.d_docGpu, data.d_docGpu + data.numDocs, data.d_rstGpuBinarySearch, CopyPredicator());
        data.rstGpuBinarySearchSize = endPtr - data.d_rstGpuBinarySearch;
        if (t >= 0)
            data.timeMsGpuBinarySearch += timer.tocMs();
    }
    data.timeMsGpuBinarySearch /= setting.numTrials;

    cout << "GPU binary search time = " << data.timeMsGpuBinarySearch << " ms" << endl;
}