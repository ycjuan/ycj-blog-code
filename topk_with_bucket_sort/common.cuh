#pragma once

struct Doc
{
    int docId;
    float score;
    bool operator==(const Doc &other) const
    {
        return docId == other.docId && score == other.score;
    }
};

inline __device__ __host__ bool scoreComparator(const Doc &a, const Doc &b)
{
    return a.score > b.score;
}

struct ScorePredicator
{
    inline __host__ __device__ bool operator()(const Doc &a, const Doc &b)
    {
        return scoreComparator(a, b);
    }
};