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

void methodCpu(Data &data, Setting setting)
{
    data.rstCpu1D.clear();
    for (int i = 0; i < data.numDocs; i++)
    {
        Doc &doc = data.doc1D[i];
        for (int j = 0; j < data.listSize; j++)
        {
            if (doc.docHash == data.list1D[j])
            {
                doc.isIn = 1;
                break;
            }
        }
        if (doc.isIn == 1)
            data.rstCpu1D.push_back(doc);
    }
}

void __global__ methodGpuNaiveKernel(Doc *d_doc, uint64_t *d_list, int numDocs, int listSize)
{
    int docIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (docIdx >= numDocs)
        return;

    for (int listIdx = 0; listIdx < listSize; listIdx++)
    {
        Doc &doc = d_doc[docIdx];
        if (doc.docHash == d_list[listIdx])
        {
            doc.isIn = 1;
            break;
        }
    }
}

struct CopyPredicator
{
    __device__ bool operator()(const Doc &doc)
    {
        return doc.isIn == 1;
    }
};

void methodGpuNaive(Data &data, Setting setting)
{
    int blockSize = 256;
    int numBlocks = (data.numDocs + blockSize - 1) / blockSize;

    CudaTimer timer;
    for (int t = -3; t < setting.numTrials; t++)
    {
        CHECK_CUDA(cudaMemcpy(data.d_docGpu, data.doc1D.data(), data.numDocs * sizeof(Doc), cudaMemcpyHostToDevice));

        timer.tic();
        methodGpuNaiveKernel<<<numBlocks, blockSize>>>(data.d_docGpu, data.d_list, data.numDocs, data.listSize);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());

        Doc *endPtr = thrust::copy_if(thrust::device, data.d_docGpu, data.d_docGpu + data.numDocs, data.d_rstGpuNaive, CopyPredicator());
        data.rstGpuNaiveSize = endPtr - data.d_rstGpuNaive;
        if (t >= 0)
            data.timeMsGpuNaive += timer.tocMs();
    }
    data.timeMsGpuNaive /= setting.numTrials;

    cout << "GPU naive time = " << data.timeMsGpuNaive << " ms" << endl;
}