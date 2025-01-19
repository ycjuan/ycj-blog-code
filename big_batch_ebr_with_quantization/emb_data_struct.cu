#include <iostream>
#include <sstream>
#include <random>

#include "emb.cuh"

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

#define CHECK_CUBLAS(func)                                                                                                  \
    {                                                                                                                       \
        cublasStatus_t status = (func);                                                                                     \
        if (status != CUBLAS_STATUS_SUCCESS)                                                                                \
        {                                                                                                                   \
            string error = "cuBLAS API failed at line " + to_string(__LINE__) + " with error: " + to_string(status) + "\n"; \
            throw runtime_error(error);                                                                                     \
        }                                                                                                                   \
    }

using namespace std;

void EmbData::initRand(int numDocs, int numReqs, int embDim, MemLayout docMemLayout, MemLayout reqMemLayout)
{
    cublasCreate(&cublasHandle);
    CHECK_CUBLAS(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

    this->numDocs = numDocs;
    this->numReqs = numReqs;
    this->embDim = embDim;
    this->docMemLayout = docMemLayout;
    this->reqMemLayout = reqMemLayout;

    CHECK_CUDA(cudaMalloc(&d_doc, (size_t)numDocs * embDim * sizeof(T_EMB)));
    CHECK_CUDA(cudaMalloc(&d_req, (size_t)numReqs * embDim * sizeof(T_EMB)));
    CHECK_CUDA(cudaMalloc(&d_rst, (size_t)numDocs * numReqs * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_doc, (size_t)numDocs * embDim * sizeof(T_EMB)));
    CHECK_CUDA(cudaMallocHost(&h_req, (size_t)numReqs * embDim * sizeof(T_EMB)));
    CHECK_CUDA(cudaMallocHost(&h_rst, (size_t)numDocs * numReqs * sizeof(float)));

    default_random_engine generator;
    uniform_real_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < numDocs * embDim; i++)
        h_doc[i] = (T_EMB)distribution(generator);
    for (int i = 0; i < numReqs * embDim; i++)
        h_req[i] = (T_EMB)distribution(generator);
    
    CHECK_CUDA(cudaMemcpy(d_doc, h_doc, (size_t)numDocs * embDim * sizeof(T_EMB), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_req, h_req, (size_t)numReqs * embDim * sizeof(T_EMB), cudaMemcpyHostToDevice));
}

void EmbData::initRandMask(float passRate)
{
    CHECK_CUDA(cudaMalloc(&d_mask, (size_t)numDocs * numReqs * sizeof(Pair)));
    CHECK_CUDA(cudaMallocHost(&h_mask, (size_t)numDocs * numReqs * sizeof(Pair)));

    default_random_engine generator;
    uniform_real_distribution<float> distribution(0.0, 1.0);
    size_t currMaskIdx = 0;
    for (int i = 0; i < numDocs; i++)
    {
        for (int r = 0; r < numReqs; r++)
        {
            float randVal = distribution(generator);
            if (randVal < passRate)
            {
                h_mask[currMaskIdx].docIdx = i;
                h_mask[currMaskIdx].reqIdx = r;
                currMaskIdx++;
            }

        }
    }
    maskSize = currMaskIdx;
    
    CHECK_CUDA(cudaMemcpy(d_mask, h_mask, (size_t)numDocs * numReqs * sizeof(Pair), cudaMemcpyHostToDevice));
}

void EmbData::print()
{
        ostringstream oss;
        oss << "numDocs: " << numDocs << ", numReqs: " << numReqs << ", embDim: " << embDim << endl;
        oss << "docMemLayout: " << (docMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "reqMemLayout: " << (reqMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        oss << "rstMemLayout: " << (rstMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
        cout << oss.str();
}

void EmbData::free()
{
    cudaFree(d_doc);
    cudaFree(d_req);
    cudaFree(d_rst);
    cudaFreeHost(h_rst);
    cublasDestroy(cublasHandle);
}