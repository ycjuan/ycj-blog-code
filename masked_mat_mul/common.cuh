#ifndef COMMON_CUH
#define COMMON_CUH

#include <sstream>
#include <cublas_v2.h>

using namespace std;

typedef half T;

enum MemLayout
{
    ROW_MAJOR,
    COL_MAJOR
};

struct Pair
{
    int reqIdx;
    int docIdx;
};

struct Data
{
    int numDocs;
    int numReqs;
    int embDim;
    int numEligibleDocs;
    T *d_doc; // M=numDocs x N=embDim
    T *d_req; // M=numReqs x N=embDim
    float *d_rst_kernel; // M=numDocs x N=numReqs
    float *d_rst_cublas; // M=numDocs x N=numReqs
    float *h_rst_cpu;
    Pair *d_eligiblePairs;
    MemLayout docMemLayout;
    MemLayout reqMemLayout;
    MemLayout rstLayoutCpu;
    MemLayout rstLayoutGpuKernel;
    MemLayout rstLayoutGpuCublas;

    void free()
    {
        cudaFree(d_doc);
        cudaFree(d_req);
        cudaFree(d_rst_kernel);
        cudaFree(d_rst_cublas);
        cudaFreeHost(h_rst_cpu);
    }

    void print()
    {
        ostringstream oss;
        oss << "numDocs: " << numDocs << ", numReqs: " << numReqs << ", embDim: " << embDim << ", numEligibleDocs: " << numEligibleDocs << endl;
        oss << "docMemLayout: " << (docMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "reqMemLayout: " << (reqMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstLayoutCpu: " << (rstLayoutCpu == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstLayoutGpuKernel: " << (rstLayoutGpuKernel == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstLayoutGpuCublas: " << (rstLayoutGpuCublas == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        cout << oss.str();
    }
};

struct Setting
{
    int numTrials;
};

inline __device__ __host__ size_t getMemAddr(int i, int j, int M, int N, MemLayout layout)
{
    if (layout == ROW_MAJOR)
        return (size_t)i * N + j;
    else
        return (size_t)j * M + i;
}

bool pairComparator(const Pair &a, const Pair &b)
{
    if (a.reqIdx != b.reqIdx)
        return a.reqIdx < b.reqIdx;
    else
        return a.docIdx < b.docIdx;
}

#endif