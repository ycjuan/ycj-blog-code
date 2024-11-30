#ifndef COMMON_CUH
#define COMMON_CUH

#include <sstream>
#include <iostream>
#include <algorithm>
#include <cublas_v2.h>

using namespace std;

//typedef __nv_bfloat16 T; // This is only supposed by CUSPARSE_FORMAT_BSR format
//typedef half T;
typedef float T;

enum MemLayout
{
    ROW_MAJOR,
    COL_MAJOR
};

struct Pair
{
    int docIdx;
    int reqIdx;
    float score;
};

struct Data
{
    int numDocs;
    int numReqs;
    int embDim;
    size_t numPairsToScore;
    T *d_doc; // M=numDocs x N=embDim
    T *d_req; // M=numReqs x N=embDim
    Pair *d_PairsToScore;    
    Pair *h_rstCpu;
    Pair *d_rstCuda;
    Pair *d_rstCusparse;
    
    MemLayout docMemLayout;
    MemLayout reqMemLayout;

    void free()
    {
        cudaFree(d_doc);
        cudaFree(d_req);
        cudaFree(d_rstCuda);
        cudaFree(d_rstCusparse);
        cudaFreeHost(h_rstCpu);
    }

    void print()
    {
        ostringstream oss;
        oss << "numDocs: " << numDocs << ", numReqs: " << numReqs << ", embDim: " << embDim << ", numPairsToScore: " << numPairsToScore << endl;
        oss << "docMemLayout: " << (docMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "reqMemLayout: " << (reqMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        cout << oss.str();
    }

    void swapDocReq()
    {
        swap(d_doc, d_req);
        swap(numDocs, numReqs);
        swap(docMemLayout, reqMemLayout);
        for (size_t i = 0; i < numPairsToScore; i++)
            swap(d_PairsToScore[i].docIdx, d_PairsToScore[i].reqIdx);
    }
};

struct Setting
{
    int numTrials;
    bool swapDocReq;
    bool reqFirst;
    MemLayout docMemLayout;
    MemLayout reqMemLayout;

    void print()
    {
        ostringstream oss;
        oss << "numTrials: " << numTrials << ", swapDocReq: " << swapDocReq << ", reqFirst: " << reqFirst << endl;
        oss << "docMemLayout: " << (docMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "reqMemLayout: " << (reqMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        cout << oss.str();
    }
};

inline __device__ __host__ size_t getMemAddr(int i, int j, int M, int N, MemLayout layout)
{
    if (layout == ROW_MAJOR)
        return (size_t)i * N + j;
    else
        return (size_t)j * M + i;
}

inline bool pairComparatorReqFirst(const Pair &a, const Pair &b)
{
    if (a.reqIdx != b.reqIdx)
        return a.reqIdx < b.reqIdx;
    else
        return a.docIdx < b.docIdx;
}

inline bool pairComparatorDocFirst(const Pair &a, const Pair &b)
{
    if (a.docIdx != b.docIdx)
        return a.docIdx < b.docIdx;
    else
        return a.reqIdx < b.reqIdx;
}

inline void coo2Csr(Data data, int *dC_offsets, int *dC_columns, float *dC_values)
{
    sort(data.d_PairsToScore, data.d_PairsToScore + data.numPairsToScore, pairComparatorDocFirst);

    Pair pair = data.d_PairsToScore[0];
    dC_offsets[0] = 0;
    dC_columns[0] = pair.reqIdx;
    dC_values[0]  = pair.score;
    int currDocIdx = pair.docIdx;
    for (size_t v = 1; v < data.numPairsToScore; v++)
    {
        pair = data.d_PairsToScore[v];
        for (int docIdx = currDocIdx; docIdx < pair.docIdx; docIdx++)
        {
            //cout << "docIdx: " << docIdx << ", currDocIdx: " << currDocIdx << ", pair.docIdx: " << pair.docIdx << ", v: " << v << endl;
            dC_offsets[docIdx + 1] = v;
        }
        currDocIdx = pair.docIdx;
        dC_columns[v] = pair.reqIdx;
        dC_values[v]  = pair.score;
    }
    for (int docIdx = currDocIdx; docIdx < data.numDocs; docIdx++)
        dC_offsets[docIdx + 1] = data.numPairsToScore;
}

#endif