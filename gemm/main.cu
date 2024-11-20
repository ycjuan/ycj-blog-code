#include <string>
#include <stdexcept>
#include <iostream>
#include <random>
#include <sstream>
#include <cublas_v2.h>
#include <type_traits>

#include "util.cuh"

// Ref: Some code snippets are borrowed from
// https://github.com/NVIDIA-developer-blog/code-samples/blob/708ce9137eb5ac7682f788e5d5b8279c7e2578ed/posts/tensor-cores/simpleTensorCoreGEMM.cu

using namespace std;

enum MemLayout
{
    ROW_MAJOR,
    COL_MAJOR
};

int kNumDocs = 1 << 20;
int kNumReqs = 1 << 0;
int kEmbDim = 1 << 10;
int kNumTrials = 100;
MemLayout kMemLayoutDoc = COL_MAJOR;
MemLayout kMemLayoutReq = ROW_MAJOR;
MemLayout kMemLayoutRstCpu = COL_MAJOR;
MemLayout kMemLayoutRstGpuKernel = COL_MAJOR;
MemLayout kMemLayoutRstGpuCublas = COL_MAJOR;
//typedef __nv_bfloat16 T;
typedef half T; 
// IMPORTANT!!! only __nv_bfloat16 and half are supported for now

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

__device__ __host__ size_t getMemAddr(int i, int j, int M, int N, MemLayout layout)
{
    if (layout == ROW_MAJOR)
        return (size_t)i * N + j;
    else
        return (size_t)j * M + i;
}

template <typename T>
struct Data
{
    int numDocs;
    int numReqs;
    int embDim;
    T *d_doc; // M=numDocs x N=embDim
    T *d_req; // M=numReqs x N=embDim
    float *d_rst_kernel; // M=numDocs x N=numReqs
    float *d_rst_cublas; // M=numDocs x N=numReqs
    float *h_rst_cpu;
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
        oss << "numDocs: " << numDocs << ", numReqs: " << numReqs << ", embDim: " << embDim << endl;
        oss << "docMemLayout: " << (docMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "reqMemLayout: " << (reqMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstLayoutCpu: " << (rstLayoutCpu == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstLayoutGpuKernel: " << (rstLayoutGpuKernel == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstLayoutGpuCublas: " << (rstLayoutGpuCublas == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        cout << oss.str();
    }
};

template <typename T>
Data<T> genData()
{
    Data<T> data;
    data.numDocs = kNumDocs;
    data.numReqs = kNumReqs;
    data.embDim = kEmbDim;
    data.docMemLayout = kMemLayoutDoc;
    data.reqMemLayout = kMemLayoutReq;
    data.rstLayoutCpu = kMemLayoutRstCpu;
    data.rstLayoutGpuKernel = kMemLayoutRstGpuKernel;
    data.rstLayoutGpuCublas = kMemLayoutRstGpuCublas;
    data.print();
    
    CHECK_CUDA(cudaMallocManaged(&data.d_doc, (size_t)data.numDocs * data.embDim * sizeof(T)));
    CHECK_CUDA(cudaMallocManaged(&data.d_req, (size_t)data.numReqs * data.embDim * sizeof(T)));
    CHECK_CUDA(cudaMallocManaged(&data.d_rst_kernel, (size_t)data.numDocs * data.numReqs * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&data.d_rst_cublas, (size_t)data.numDocs * data.numReqs * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&data.h_rst_cpu, (size_t)data.numDocs * data.numReqs * sizeof(float)));

    default_random_engine generator;
    uniform_real_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < data.numDocs * data.embDim; i++)
        data.d_doc[i] = (T)distribution(generator);
    for (int i = 0; i < data.numReqs * data.embDim; i++)
        data.d_req[i] = (T)distribution(generator);

    return data;
}

template <typename T>
void checkData(Data<T> data)
{
    for (int i = 0; i < data.numDocs; i++)
    {
        for (int j = 0; j < data.numReqs; j++)
        {
            float cpuVal = data.h_rst_cpu[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutCpu)];
            float gpuKernelVal = data.d_rst_kernel[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutGpuKernel)];
            float gpuCublasVal = data.d_rst_cublas[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutGpuCublas)];

            if (abs(cpuVal - gpuKernelVal) / abs(gpuKernelVal) > 1e-3)
            {
                cout << "Kernel error at (" << i << ", " << j << "): " << cpuVal << " != " << gpuKernelVal << endl;
                return;
            }
            
            if (abs(cpuVal - gpuCublasVal) / abs(gpuKernelVal) > 1e-3)
            {
                cout << "Cublas error at (" << i << ", " << j << "): " << cpuVal << " != " << gpuCublasVal << endl;
                return;
            }
        }
    }
}

template <typename T>
void matMulCublas(Data<T> data)
{
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    float alpha = 1.0;
    float beta = 0.0;

    int MATRIX_M = data.numDocs;
    int MATRIX_N = data.numReqs;
    int MATRIX_K = data.embDim;
    T *a_fp16 = data.d_doc;
    T *b_fp16 = data.d_req;
    float *c_cublas = data.d_rst_cublas;

    cublasOperation_t trana = (data.docMemLayout == COL_MAJOR) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t tranb = (data.reqMemLayout == COL_MAJOR) ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = (data.docMemLayout == COL_MAJOR) ? MATRIX_M : MATRIX_K;
    int ldb = (data.reqMemLayout == COL_MAJOR) ? MATRIX_N : MATRIX_K;
    cudaDataType aType = (is_same<T, half>::value) ? CUDA_R_16F : CUDA_R_16BF;
    cudaDataType bType = (is_same<T, half>::value) ? CUDA_R_16F : CUDA_R_16BF;

    CudaTimer timer;
    for (int t = -3; t < kNumTrials; t++)
    {
        if (t == 0)
            timer.tic();
        cublasErrCheck(cublasGemmEx(cublasHandle, trana, tranb,
                                    MATRIX_M, MATRIX_N, MATRIX_K,
                                    &alpha,
                                    a_fp16, aType, lda,
                                    b_fp16, bType, ldb,
                                    &beta,
                                    c_cublas, CUDA_R_32F, MATRIX_M,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    cout << "Cublas time: " << timer.tocMs() / kNumTrials << " ms" << endl;

    cublasDestroy(cublasHandle);
}

template<typename T>
void matMulCpu(Data<T> data)
{
    Timer timer;
    timer.tic();
    #pragma omp parallel for
    for (int i = 0; i < data.numDocs; i++)
    {
        for (int j = 0; j < data.numReqs; j++)
        {
            float sum = 0;
            for (int k = 0; k < data.embDim; k++)
            {
                T reqVal = data.d_req[getMemAddr(j, k, data.numReqs, data.embDim, data.reqMemLayout)];
                T docVal = data.d_doc[getMemAddr(i, k, data.numDocs, data.embDim, data.docMemLayout)];
                sum += (float)reqVal * (float)docVal;
            }
            data.h_rst_cpu[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutCpu)] = (half)sum;
        }
    }
    cout << "CPU time: " << timer.tocMs() << " ms" << endl;
}

template <typename T>
__global__ void matMul(Data<T> data)
{
    int threadId = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int i = threadId / data.numReqs;
    int j = threadId % data.numReqs;

    if (i < data.numDocs && j < data.numReqs)
    {
        float sum = 0;
        for (int k = 0; k < data.embDim; k++)
        {
            T reqVal = data.d_req[getMemAddr(j, k, data.numReqs, data.embDim, data.reqMemLayout)];
            T docVal = data.d_doc[getMemAddr(i, k, data.numDocs, data.embDim, data.docMemLayout)];            
            sum += float(reqVal * docVal);
        }
        data.d_rst_kernel[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutGpuKernel)] = sum;
    }
}

template <typename T>
void matMulKernel(Data<T> data)
{
    int blockSize = 512;
    int gridSize = size_t(data.numDocs) * data.numReqs / blockSize;
    CudaTimer timer;
    for (int t = -3; t < kNumTrials; t++)
    {
        if (t == 0)
            timer.tic();
        matMul<<<gridSize, blockSize>>>(data);
        cudaDeviceSynchronize();
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess)
        {
            ostringstream oss;
            oss << "Kernel launch failed with error: " << cudaGetErrorString(status) << "\n";
            throw runtime_error(oss.str());
        }
    }
    cout << "Kernel time: " << timer.tocMs() / kNumTrials << " ms" << endl;
}

int main()
{
    Data<T> data = genData<T>();

    matMulCublas(data);
    matMulKernel(data);
    matMulCpu(data);

    checkData(data);

    data.free();

    return 0;
}