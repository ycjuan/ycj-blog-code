#include <random>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cuda_bf16.h>

#include "util.cuh"
#include "core.cuh"

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

float computeRMSE(const vector<Doc> &v_rstA, const vector<Doc> &v_rstB)
{
    if (v_rstA.size() != v_rstB.size())
        throw runtime_error("v_rstA.size() != v_rstB.size()");

    double squaredErrorSum = 0;
    for (int i = 0; i < v_rstA.size(); i++)
    {
        double diff = (v_rstA[i].score - v_rstB[i].score) / v_rstA[i].score; // divide by v_rstA[i] so this error means "how much off from v_rstA[i]"
        squaredErrorSum += diff * diff;
    }
    double rmse = sqrt(squaredErrorSum / v_rstA.size());
    return rmse;
}

void genRandEmbFP32(float *d_emb, int numDocs, int embDim)
{
    default_random_engine generator;
    uniform_real_distribution<float> distribution(-1.0, 1.0);
    for (int i = 0; i < numDocs; i++)
    {
        double normSum = 0;
        for (int j = 0; j < embDim; j++)
        {
            float &v = d_emb[getMemAddr(i, j, numDocs, embDim)];
            v = distribution(generator);
            normSum += v * v;
        }
        double normalizer = 1.0 / sqrt(normSum);
        for (int j = 0; j < embDim; j++)
            d_emb[getMemAddr(i, j, numDocs, embDim)] *= normalizer;
    }
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
    cout << "Running experiment with numDocs=" << numDocs << ", embDim=" << embDim << ", density=" << density << endl;

    int numActiveDocs = (int)(numDocs * density);

    float *d_docEmb_fp32 = nullptr;
    float *d_reqEmb_fp32 = nullptr;
    Doc *d_doc = nullptr;
    CHECK_CUDA(cudaMallocManaged(&d_docEmb_fp32, numDocs * embDim * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&d_reqEmb_fp32, embDim * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&d_doc, numActiveDocs * sizeof(Doc)));
    genRandEmbFP32(d_docEmb_fp32, numDocs, embDim);
    genRandEmbFP32(d_reqEmb_fp32, 1, embDim);
    genRandActiveDocs(d_doc, numDocs, numActiveDocs);

    __nv_bfloat16 *d_docEmb_bf16 = nullptr;
    __nv_bfloat16 *d_reqEmb_bf16 = nullptr;
    CHECK_CUDA(cudaMallocManaged(&d_docEmb_bf16, numDocs * embDim * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMallocManaged(&d_reqEmb_bf16, embDim * sizeof(__nv_bfloat16)));
    copyAsBF16(d_docEmb_fp32, d_docEmb_bf16, numDocs, embDim);
    copyAsBF16(d_reqEmb_fp32, d_reqEmb_bf16, 1, embDim);

    float timeMs, rmse;

    cout << "data type = fp32, accumulator type = fp64, ";
    timeMs = 0;
    for (int t = -3; t < kNumTrials; t++)
    {
        float timeMs1 = score_fp32_bf16<float, double>(d_docEmb_fp32, d_reqEmb_fp32, d_doc, numDocs, numActiveDocs, embDim);
        if (t >= 0)
            timeMs += timeMs1;
    }
    timeMs /= kNumTrials;
    vector<Doc> v_doc_fp32_fp64(numActiveDocs);
    CHECK_CUDA(cudaMemcpy(v_doc_fp32_fp64.data(), d_doc, numActiveDocs * sizeof(Doc), cudaMemcpyDeviceToHost));
    cout << "time = " << timeMs << " ms" << endl;

    cout << "data type = fp32, accumulator type = fp32, ";
    timeMs = 0;
    for (int t = -3; t < kNumTrials; t++)
    {
        float timeMs1 = score_fp32_bf16<float, float>(d_docEmb_fp32, d_reqEmb_fp32, d_doc, numDocs, numActiveDocs, embDim);
        if (t >= 0)
            timeMs += timeMs1;
    }
    timeMs /= kNumTrials;
    vector<Doc> v_doc_fp32_fp32(numActiveDocs);
    CHECK_CUDA(cudaMemcpy(v_doc_fp32_fp32.data(), d_doc, numActiveDocs * sizeof(Doc), cudaMemcpyDeviceToHost));
    rmse = computeRMSE(v_doc_fp32_fp64, v_doc_fp32_fp32);
    cout << "time = " << timeMs << " ms" << ", rmse = " << rmse << endl;


    cout << "data type = bf16, accumulator type = fp64, ";
    timeMs = 0;
    for (int t = -3; t < kNumTrials; t++)
    {
        float timeMs1 = score_fp32_bf16<__nv_bfloat16, double>(d_docEmb_bf16, d_reqEmb_bf16, d_doc, numDocs, numActiveDocs, embDim);
        if (t >= 0)
            timeMs += timeMs1;
    }
    timeMs /= kNumTrials;
    vector<Doc> v_doc_bf16_fp64(numActiveDocs);
    CHECK_CUDA(cudaMemcpy(v_doc_bf16_fp64.data(), d_doc, numActiveDocs * sizeof(Doc), cudaMemcpyDeviceToHost));
    rmse = computeRMSE(v_doc_fp32_fp64, v_doc_bf16_fp64);
    cout << "time = " << timeMs << " ms" << ", rmse = " << rmse << endl;

    cout << "data type = bf16, accumulator type = fp32, ";
    timeMs = 0;
    for (int t = -3; t < kNumTrials; t++)
    {
        float timeMs1 = score_fp32_bf16<__nv_bfloat16, float>(d_docEmb_bf16, d_reqEmb_bf16, d_doc, numDocs, numActiveDocs, embDim);
        if (t >= 0)
            timeMs += timeMs1;
    }
    timeMs /= kNumTrials;
    vector<Doc> v_doc_bf16_fp32(numActiveDocs);
    CHECK_CUDA(cudaMemcpy(v_doc_bf16_fp32.data(), d_doc, numActiveDocs * sizeof(Doc), cudaMemcpyDeviceToHost));
    rmse = computeRMSE(v_doc_fp32_fp64, v_doc_bf16_fp32);
    cout << "time = " << timeMs << " ms" << ", rmse = " << rmse << endl;

    cout << "data type = bf16, accumulator type = bf16, ";
    timeMs = 0;
    for (int t = -3; t < kNumTrials; t++)
    {
        float timeMs1 = score_fp32_bf16<__nv_bfloat16, __nv_bfloat16>(d_docEmb_bf16, d_reqEmb_bf16, d_doc, numDocs, numActiveDocs, embDim);
        if (t >= 0)
            timeMs += timeMs1;
    }
    timeMs /= kNumTrials;
    vector<Doc> v_doc_bf16_bf16(numActiveDocs);
    CHECK_CUDA(cudaMemcpy(v_doc_bf16_bf16.data(), d_doc, numActiveDocs * sizeof(Doc), cudaMemcpyDeviceToHost));
    rmse = computeRMSE(v_doc_fp32_fp64, v_doc_bf16_bf16);
    cout << "time = " << timeMs << " ms" << ", rmse = " << rmse << endl;
}

int main()
{
    runExp(100000, 128, 0.5);

    return 0;
}