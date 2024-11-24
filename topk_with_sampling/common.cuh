#ifndef COMMON_CUH
#define COMMON_CUH

struct Pair
{
    int reqId;
    int docId;
    float score;
    bool operator==(const Pair &other) const
    {
        return docId == other.docId && score == other.score && reqId == other.reqId;
    }
};

inline __device__ __host__ bool scoreComparator(const Pair &a, const Pair &b)
{
    return a.score > b.score;
}

struct ScorePredicator
{
    inline __host__ __device__ bool operator()(const Pair &a, const Pair &b)
    {
        return scoreComparator(a, b);
    }
};

inline __device__ __host__ size_t getMemAddr(int reqId, int docId, size_t numDocs)
{
    return reqId * numDocs + docId;
}

#endif // COMMON_CUH