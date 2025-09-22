#ifndef BATCH_SCALABILITY_DATA_STRUCT_CUH
#define BATCH_SCALABILITY_DATA_STRUCT_CUH

#include <cstdint>

namespace BatchScalability
{

struct Data
{
    int numReqs;
    int numDocs;
    int embDim;
    float* d_docData = nullptr;
    float* d_reqData = nullptr;
    float* h_rstDataCpu = nullptr;
    float* d_rstDataGpu = nullptr;
};

__device__ __host__ inline uint64_t getMemAddrRowMajor(int rowIdx, int colIdx, int numRows, int numCols)
{
    return (uint64_t)(rowIdx * numCols + colIdx);
}

__device__ __host__ inline uint64_t getMemAddrColMajor(int rowIdx, int colIdx, int numRows, int numCols)
{
    return (uint64_t)(colIdx * numRows + rowIdx);
}

__device__ __host__ inline uint64_t getMemAddrReq(int reqIdx, int embIdx, int numReqs, int numDims)
{
    return getMemAddrRowMajor(reqIdx, embIdx, numReqs, numDims);
}

__device__ __host__ inline uint64_t getMemAddrDoc(int docIdx, int embIdx, int numDocs, int numDims)
{
    return getMemAddrRowMajor(docIdx, embIdx, numDocs, numDims);
}

__device__ __host__ inline uint64_t getMemAddrRst(int reqIdx, int docIdx, int numReqs, int numDocs)
{
    return getMemAddrRowMajor(reqIdx, docIdx, numReqs, numDocs);
}

Data genData(int numReqs, int numDocs, int embDim);

void freeData(Data& data);

} // namespace BatchScalability

#endif