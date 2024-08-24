#include <random>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cuda_bf16.h>

#include "util.cuh"
#include "core.cuh"

const int kNumDocs = 1000000;
const int kEmbDim = 128;
const int kNumTrials = 10;

using namespace std;

#define CHECK_CUDA(func)                                                                                                                     \
    {                                                                                                                                        \
        cudaError_t status = (func);                                                                                                         \
        if (status != cudaSuccess)                                                                                                           \
        {                                                                                                                                    \
            string error = "[main.cu] CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                                      \
        }                                                                                                                                    \
    }

bool compareRst(const vector<float> &v_rstA, const vector<float> &v_rstB)
{
    if (v_rstA.size() != v_rstB.size())
        return false;
    for (int i = 0; i < v_rstA.size(); i++)
    {
        if (abs(v_rstA[i] - v_rstB[i]) > 1e-5)
            return false;
    }
    return true;
}

void genRandEmbFP32(float *d_emb, int numDocs, int embDim)
{
    default_random_engine generator;
    uniform_real_distribution<float> distribution(-1.0, 1.0);
    vector<float> v_emb(numDocs * embDim);
    for (auto &v : v_emb)
        v = distribution(generator);
    for (int i = 0; i < numDocs; i++)
    {
        double normSum = 0;
        for (int j = 0; j < embDim; j++)
        {
            float v = v_emb[getMemAddr(i, j, numDocs, embDim)];
            normSum += v * v;
        }
        double normalizer = 1.0 / sqrt(normSum);
        for (int j = 0; j < embDim; j++)
            v_emb[getMemAddr(i, j, numDocs, embDim)] *= normalizer;
    }
    CHECK_CUDA(cudaMemcpy(d_emb, v_emb.data(), numDocs * embDim * sizeof(float), cudaMemcpyHostToDevice));
}

void copyAsBF16(float *d_fp32, __nv_bfloat16 *d_bf16, int numDocs, int embDim)
{
    for (int i = 0; i < numDocs; i++)
        for (int j = 0; j < embDim; j++)
            d_bf16[getMemAddr(i, j, numDocs, embDim)] = (__nv_bfloat16)d_fp32[getMemAddr(i, j, numDocs, embDim)];
}

void genRandActiveDocs(Doc *d_doc, int numDocs, int numActiveDocs)
{
    vector<Doc> v_doc(numDocs);
    for (int i = 0; i < numDocs; i++)
    {
        v_doc[i].docIdx = i;
        v_doc[i].score = 0;
    }
    shuffle(v_doc.begin(), v_doc.end(), default_random_engine());
    assert(v_doc[0].docIdx != 0);
    assert(v_doc[numDocs-1].docIdx != numDocs-1);
    v_doc.resize(numActiveDocs);
    CHECK_CUDA(cudaMemcpy(d_doc, v_doc.data(), numActiveDocs * sizeof(Doc), cudaMemcpyHostToDevice));
}

void runExp(int numDocs, int embDim, float density)
{
    int numActiveDocs = (int)(numDocs * density);

    float *d_docEmb_fp32 = nullptr;
    float *d_reqEmb_fp32 = nullptr;
    Doc *d_doc = nullptr;
    CHECK_CUDA(cudaMalloc(&d_docEmb_fp32, numDocs * embDim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_reqEmb_fp32, embDim * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&d_doc, numActiveDocs * sizeof(Doc)));
    genRandEmbFP32(d_docEmb_fp32, numDocs, embDim);
    genRandEmbFP32(d_reqEmb_fp32, 1, embDim);
    genRandActiveDocs(d_doc, numDocs, numActiveDocs);

}

int main()
{
}