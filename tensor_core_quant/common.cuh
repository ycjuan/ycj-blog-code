#ifndef COMMON_CUH
#define COMMON_CUH

#include <sstream>
#include <iostream>

using namespace std;

// IMPORTANT: Do not change the following typedefs. These are only ones that works for WMMA.
typedef uint32_t T_QUANT;
typedef int32_t T_RST;

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
    int numInt;
    T_QUANT *d_doc; // M=numDocs x N=numInt64
    T_QUANT *d_req; // M=numReqs x N=numInt64
    T_RST *d_rst_kernel; // M=numDocs x N=numReqs
    T_RST *d_rstWmmaSimple; // M=numDocs x N=numReqs
    T_RST *d_rstWmmaUnroll; // M=numDocs x N=numReqs
    T_RST *h_rst_cpu;
    MemLayout docMemLayout;
    MemLayout reqMemLayout;
    MemLayout rstLayoutCpu;
    MemLayout rstLayoutGpuCuda;
    MemLayout rstLayoutGpuWmmaSimple;
    MemLayout rstLayoutGpuWmmaUnroll;

    void free()
    {
        cudaFree(d_doc);
        cudaFree(d_req);
        cudaFree(d_rst_kernel);
        cudaFree(d_rstWmmaSimple);
        cudaFree(d_rstWmmaUnroll);
        cudaFreeHost(h_rst_cpu);
    }

    void print()
    {
        ostringstream oss;
        oss << "numDocs: " << numDocs << ", numReqs: " << numReqs << ", numInt: " << numInt << ", numBits: " << sizeof(T_QUANT) * 8 << endl;
        oss << "docMemLayout: " << (docMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "reqMemLayout: " << (reqMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstLayoutCpu: " << (rstLayoutCpu == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstLayoutGpuCuda: " << (rstLayoutGpuCuda == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstLayoutGpuWmmaSimple: " << (rstLayoutGpuWmmaSimple == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstLayoutGpuWmmaUnroll: " << (rstLayoutGpuWmmaUnroll == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
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
