#ifndef CORE_CUH
#define CORE_CUH

#include <string>

using namespace std;

inline __host__ __device__ int getMemAddr(int docIdx, int embIdx, int numDocs, int embDim)
{
    //return embIdx * numDocs + docIdx;
    return docIdx * embDim + embIdx;
}

struct Doc
{
    int docIdx;
    float score;
};

const int kBlockSize = 256;

#define CHECK_CUDA(func)                                                                                                                     \
    {                                                                                                                                        \
        cudaError_t status = (func);                                                                                                         \
        if (status != cudaSuccess)                                                                                                           \
        {                                                                                                                                    \
            string error = "[main.cu] CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                                      \
        }                                                                                                                                    \
    }

template <typename T_EMB, typename T_ACC>
__global__ void kernel_fp32_bf16(T_EMB *d_docEmb, T_EMB *d_reqEmb, Doc *d_doc, int numAllDocs, int numActiveDocs, int embDim)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < numActiveDocs)
    {
        Doc &doc = d_doc[d];
        T_ACC acc = 0;
        int i = doc.docIdx;
        for (int j = 0; j < embDim; j++)
        {
            T_EMB docVal = d_docEmb[getMemAddr(i, j, numAllDocs, embDim)];
            T_EMB reqVal = d_reqEmb[getMemAddr(0, j, 1, embDim)];
            acc += (T_ACC)(docVal * reqVal);
        }
        doc.score = (float)acc;
    }
}

template <typename T_ACC>
__global__ void kernel_bf162(__nv_bfloat162 *d_docEmb, __nv_bfloat162 *d_reqEmb, Doc *d_doc, int numAllDocs, int numActiveDocs, int embDim2)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < numActiveDocs)
    {
        Doc &doc = d_doc[d];
        T_ACC acc = 0;
        int i = doc.docIdx;
        for (int j = 0; j < embDim2; j++)
        {
            __nv_bfloat162 docVal = d_docEmb[getMemAddr(i, j, numAllDocs, embDim2)];
            __nv_bfloat162 reqVal = d_reqEmb[getMemAddr(0, j, 1, embDim2)];
            __nv_bfloat162 rst2 = docVal * reqVal;
            __nv_bfloat16 rst = rst2.x + rst2.y;
            float rst_f = __bfloat162float(rst);
            acc += (T_ACC)(rst_f);
        }
        doc.score = (float)acc;
    }
}

template <typename T_ACC>
__global__ void kernel_float4(float4 *d_docEmb, float4 *d_reqEmb, Doc *d_doc, int numAllDocs, int numActiveDocs, int embDim4)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < numActiveDocs)
    {
        Doc &doc = d_doc[d];
        T_ACC acc = 0;
        int i = doc.docIdx;
        for (int j4 = 0; j4 < embDim4; j4++)
        {
            float4 docVal = d_docEmb[getMemAddr(i, j4, numAllDocs, embDim4)];
            float4 reqVal = d_reqEmb[getMemAddr(0, j4, 1, embDim4)];
            float4 rst;
            rst.x = docVal.x * reqVal.x;
            rst.y = docVal.y * reqVal.y;
            rst.z = docVal.z * reqVal.z;
            rst.w = docVal.w * reqVal.w;
            acc += (T_ACC)(rst.x + rst.y + rst.z + rst.w);
        }
        doc.score = (float)acc;
    }
}

template <typename T_EMB, typename T_ACC>
float score_fp32_bf16(T_EMB *d_docEmb, T_EMB *d_reqEmb, Doc *d_doc, int numAllDocs, int numActiveDocs, int embDim)
{
    CudaTimer timer;
    timer.tic();
    int gridSize = (int)ceil((double)numActiveDocs / kBlockSize);
    kernel_fp32_bf16<T_EMB, T_ACC><<<gridSize, kBlockSize>>>(d_docEmb, d_reqEmb, d_doc, numAllDocs, numActiveDocs, embDim);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
    return timer.tocMs();
}

template <typename T_ACC>
float score_bf162(__nv_bfloat162 *d_docEmb, __nv_bfloat162 *d_reqEmb, Doc *d_doc, int numAllDocs, int numActiveDocs, int embDim2)
{
    CudaTimer timer;
    timer.tic();
    int gridSize = (int)ceil((double)numActiveDocs / kBlockSize);
    kernel_bf162<T_ACC><<<gridSize, kBlockSize>>>(d_docEmb, d_reqEmb, d_doc, numAllDocs, numActiveDocs, embDim2);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
    return timer.tocMs();
}


template <typename T_ACC>
float score_float4(float4 *d_docEmb, float4 *d_reqEmb, Doc *d_doc, int numAllDocs, int numActiveDocs, int embDim4)
{
    CudaTimer timer;
    timer.tic();
    int gridSize = (int)ceil((double)numActiveDocs / kBlockSize);
    kernel_float4<T_ACC><<<gridSize, kBlockSize>>>(d_docEmb, d_reqEmb, d_doc, numAllDocs, numActiveDocs, embDim4);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
    return timer.tocMs();
}

#endif // CORE_CUH