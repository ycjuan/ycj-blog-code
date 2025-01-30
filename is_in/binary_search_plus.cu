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

void __global__ methodGpuBitTrickKernel(Doc *d_doc, uint64_t *d_list, int numDocs, int listSize)
{
    int docIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (docIdx >= numDocs)
        return;

    uint64_t docHash = d_doc[docIdx].docHash;
    uint64_t andHash = ~0;
    uint64_t orHash = 0;
    for (int listIdx = 0; listIdx < listSize; listIdx++)
    {
        if (docIdx < 100)
            printf("docIdx = %d, listIdx = %d, andHash = %llu, orHash = %llu\n", docIdx, listIdx, andHash, orHash);
        andHash &= d_list[listIdx];
        orHash |= d_list[listIdx];
    }
    
    uint64_t andXorHash = andHash ^ docHash;
    uint64_t orAndHash = orHash & docHash;

    if (andHash & andXorHash != 0)
    {
        d_doc[docIdx].isIn = 0;
        return;
    }

    if (orAndHash != docHash)
    {
        d_doc[docIdx].isIn = 0;
        return;
    }

    d_doc[docIdx].isIn = 1;
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

void methodGpuBitTrick(Data &data, Setting setting)
{
    int blockSize = 256;
    int numBlocks = (data.numDocs + blockSize - 1) / blockSize;

    CudaTimer timer;
    for (int t = -3; t < setting.numTrials; t++)
    {
        CHECK_CUDA(cudaMemcpy(data.d_docGpu, data.doc1D.data(), data.numDocs * sizeof(Doc), cudaMemcpyHostToDevice));

        timer.tic();
        methodGpuBitTrickKernel<<<numBlocks, blockSize>>>(data.d_docGpu, data.d_list, data.numDocs, data.listSize);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());

        Doc *endPtr = thrust::copy_if(thrust::device, data.d_docGpu, data.d_docGpu + data.numDocs, data.d_rstGpuBitTrick, CopyPredicator());
        data.rstGpuBitTrickSize = endPtr - data.d_rstGpuBitTrick;
        cout << "pass rate = " << data.rstGpuBitTrickSize * 1.0 / data.numDocs << endl;
        if (t >= 0)
            data.timeMsGpuBitTrick += timer.tocMs();
    }
    data.timeMsGpuBitTrick /= setting.numTrials;

    cout << "GPU bit trick time = " << data.timeMsGpuBitTrick << " ms" << endl;
}

/*
vector<uint64_t> genBitTrickList(uint64_t *d_list, int listSize, int numLayers)
{
    vector<uint64_t> bitTrickList;
    for (int layerIdx = 0; layerIdx < numLayers; layerIdx++)
    {
        int groupSize = listSize >> layerIdx;
        for (int groupStartIdx = 0; groupStartIdx < listSize; groupStartIdx += groupSize)
        {
            int groupEndIdx = min(groupStartIdx + groupSize, listSize);
            uint64_t groupAnd = ~0;
            for (int i = groupStartIdx; i < groupEndIdx; i++)
                groupAnd &= d_list[i];
            uint64_t groupOr = 0;
            for (int i = groupStartIdx; i < groupEndIdx; i++)
                groupOr |= d_list[i];
            bitTrickList.push_back(groupAnd);
            bitTrickList.push_back(groupOr);
        }
    }
}

void __global__ bitTrickKernel(Doc *d_doc, uint64_t *d_bitTrickList, int numDocs, int numLayers)
{
    int docIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (docIdx >= numDocs)
        return;

    Doc &doc = d_doc[docIdx];
    doc.isIn = true;

    int currIdx = 0;
    for (int layerIdx = 0; layerIdx < numLayers; layerIdx++)
    {
        int groupSize = 1 << layerIdx;
        uint64_t tmpAnd = doc.docHash;
        uint64_t tmpOr = doc.docHash;
        for (int groupIdx; groupIdx < groupSize; groupIdx++)
        {
            tmpAnd &= d_bitTrickList[currIdx++];
            tmpOr |= d_bitTrickList[currIdx++];
        }
        tmpAnd ^= doc.docHash;
        tmpOr ^= doc.docHash;
        int numOnes = __popcll(tmpAnd);
        int numZeros = __popcll(~tmpOr);
        int numOnesMin = __popcll(doc.docHash);
        int numZerosMax = 64 - numOnesMin;
    }
}

void methodGpuBinarySearchPlus(Data &data, Setting setting)
{
    vector<uint64_t> bitTrickList = genBitTrickList(data.d_list, data.listSize, setting.numBitTrickLayers);
    uint64_t *d_bitTrickList;
    CHECK_CUDA(cudaMalloc(&d_bitTrickList, bitTrickList.size() * sizeof(uint64_t)));
    CHECK_CUDA(cudaMemcpy(d_bitTrickList, bitTrickList.data(), bitTrickList.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

}
*/