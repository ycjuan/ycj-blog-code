#ifndef COMMON_CUH
#define COMMON_CUH

struct Pair
{
    int reqIdx;
    int docIdx;
    float score;
    bool operator==(const Pair &other) const
    {
        return docIdx == other.docIdx && score == other.score && reqIdx == other.reqIdx;
    }
};

enum MemLayout
{
    ROW_MAJOR,
    COL_MAJOR
};


// Note: if-else has perf issue, but we don't use this function in performance-critical function, so it's fine
inline __device__ __host__ size_t getMemAddr(int i, int j, int M, int N, MemLayout layout)
{
    if (layout == ROW_MAJOR)
        return (size_t)i * N + j;
    else
        return (size_t)j * M + i;
}

#endif // COMMON_CUH