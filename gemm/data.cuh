#ifndef DATA_CUH
#define DATA_CUH

#include <iostream>
#include <random>

using namespace std;

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

typedef half T; 
// IMPORTANT!!! only __nv_bfloat16 and half are supported for now

enum MemLayout
{
    ROW_MAJOR,
    COL_MAJOR
};

// Important: Having if-else here is going to cause warp divergence problem.
// For performance-critical code, please write another hard-coded version (without if-else).
__device__ __host__ size_t getMemAddr(int i, int j, int M, int N, MemLayout layout)
{
    if (layout == ROW_MAJOR)
        return (size_t)i * N + j;
    else
        return (size_t)j * M + i;
}

struct Data
{
    // Meta data
    int numDocs;
    int numReqs;
    int embDim;
    int hiddenDim;

    // Read embeddings
    T *d_doc; // numDocs x embDim
    T *d_req; // numReqs x embDim
    T *d_wa; // embDim x hiddenDim
    T *d_wb; // hiddenDim x 1

    // Write results
    float *h_rst_dp_cpu; // numDocs x numReqs
    float *d_rst_dp_gpu_naive; // numDocs x numReqs
    float *d_rst_dp_gpu_cublas; // numDocs x numReqs
    float *h_rst_mlp_cpu; // numDocs x numReqs
    float *d_rst_mlp_gpu; // numDocs x numReqs

    // Memory layouts
    MemLayout docMemLayout = ROW_MAJOR;
    MemLayout reqMemLayout = ROW_MAJOR;
    MemLayout waMemLayout = ROW_MAJOR;
    MemLayout wbMemLayout = ROW_MAJOR;
    MemLayout rstDpCpuMemLayout = COL_MAJOR;
    MemLayout rstDpGpuNaiveMemLayout = ROW_MAJOR;
    MemLayout rstDpGpuCublasMemLayout = COL_MAJOR; // The output matrix of cublas is always in column-major order
    MemLayout rstMlpCpuMemLayout = COL_MAJOR;
    MemLayout rstMlpGpuMemLayout = COL_MAJOR;

    void free()
    {
        cudaFree(d_doc);
        cudaFree(d_req);
        cudaFree(d_wa);
        cudaFree(d_wb);
        cudaFree(d_rst_dp_gpu_cublas);
        cudaFree(d_rst_mlp_gpu);
        cudaFree(d_rst_dp_gpu_naive);
        cudaFreeHost(h_rst_dp_cpu);
        cudaFreeHost(h_rst_mlp_cpu);
    }

    void print()
    {
        ostringstream oss;
        oss << "numDocs: " << numDocs << ", numReqs: " << numReqs << ", embDim: " << embDim << endl;
        oss << "docMemLayout: " << (docMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "reqMemLayout: " << (reqMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "waMemLayout: " << (waMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "wbMemLayout: " << (wbMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstDpCpuMemLayout: " << (rstDpCpuMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstDpGpuNaiveMemLayout: " << (rstDpGpuNaiveMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstDpGpuCublasMemLayout: " << (rstDpGpuCublasMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstMlpCpuMemLayout: " << (rstMlpCpuMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstMlpGpuMemLayout: " << (rstMlpGpuMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        cout << oss.str();
    }
};

#endif