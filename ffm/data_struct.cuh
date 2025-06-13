#ifndef DATA_STRUCT_CUH
#define DATA_STRUCT_CUH

#include <cuda_bf16.h>

typedef __nv_bfloat16 EMB_T;

struct FFMData
{
    int embDimPerField;
    int numFields;
    int numRows;

    EMB_T* d_embData;

    __device__ __host__ size_t getMemAddr(size_t rowIdx, size_t fieldIdx, size_t embIdx)
    {
        size_t idx0 = rowIdx;
        size_t idx1 = fieldIdx;
        size_t idx2 = embIdx;

        size_t offset0 = numFields * embDimPerField;
        size_t offset1 = embDimPerField;

        return idx0 * offset0 + idx1 * offset1 + idx2;
    }
};

struct ScoringTask
{
    int reqIdx;
    int docIdx;
    float result;
};

struct ScoringTasks
{
    int numTasks;
    ScoringTask* d_tasks;
};

#endif // DATA_STRUCT_CUH