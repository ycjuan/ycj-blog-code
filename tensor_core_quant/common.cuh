#ifndef COMMON_CUH
#define COMMON_CUH

#include <sstream>
#include <iostream>

using namespace std;

typedef uint64_t T1;
typedef uint16_t T2;

enum MemLayout
{
    ROW_MAJOR,
    COL_MAJOR
};

struct Setting
{
    int kNumTrials;
};

struct Data
{
    int numDocs;
    int numReqs;
    int numInt64;
    T1 *d_doc; // M=numDocs x N=numInt64
    T1 *d_req; // M=numReqs x N=numInt64
    T2 *d_rst_kernel; // M=numDocs x N=numReqs
    T2 *d_rst_cublas; // M=numDocs x N=numReqs
    T2 *h_rst_cpu;
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
        oss << "numDocs: " << numDocs << ", numReqs: " << numReqs << ", numInt64: " << numInt64 << endl;
        oss << "docMemLayout: " << (docMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "reqMemLayout: " << (reqMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstLayoutCpu: " << (rstLayoutCpu == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstLayoutGpuKernel: " << (rstLayoutGpuKernel == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstLayoutGpuCublas: " << (rstLayoutGpuCublas == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
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

#endif