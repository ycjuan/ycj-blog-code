#ifndef CORE_CUH
#define CORE_CUH

inline __host__ __device__ u_int64_t getMemAddr(int docIdx, int embIdx, int numDocs, int embDim)
{
    return (u_int64_t)embIdx * numDocs + docIdx;
}

struct Doc
{
    int docIdx;
    float score;
};

#endif // CORE_CUH