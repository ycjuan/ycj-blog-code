#include <random>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <iomanip>
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

double computeRMSE(const vector<Doc> &v_rstA, const vector<Doc> &v_rstB)
{
    if (v_rstA.size() != v_rstB.size())
        throw runtime_error("v_rstA.size() != v_rstB.size()");

    double squaredErrorSum = 0;
    double squaredMeanSum = 0;
    for (int i = 0; i < v_rstA.size(); i++)
    {
        double diff = (v_rstA[i].score - v_rstB[i].score);
        squaredErrorSum += diff * diff;
        squaredMeanSum += v_rstA[i].score * v_rstA[i].score;
    }
    double rmse = sqrt(squaredErrorSum / v_rstA.size());
    double rms = sqrt(squaredMeanSum / v_rstA.size());
    return rmse / rms;
}

vector<vector<float>> genRandEmb(int numDocs, int embDim)
{
    default_random_engine generator;
    uniform_real_distribution<float> distribution(-1.0, 1.0);
    vector<vector<float>> emb2D(numDocs, vector<float>(embDim));
    for (int i = 0; i < numDocs; i++)
    {
        double normSum = 0;
        for (int j = 0; j < embDim; j++)
        {
            float &v = emb2D[i][j];
            v = distribution(generator);
            normSum += v * v;
        }
        double normalizer = 1.0 / sqrt(normSum);
        for (int j = 0; j < embDim; j++)
            emb2D[i][j] *= normalizer;
    }
    return emb2D;
}

void copyAsFP32(const vector<vector<float>> &emb2D, float *d_fp32, int numDocs, int embDim)
{
    for (int i = 0; i < numDocs; i++)
        for (int j = 0; j < embDim; j++)
            d_fp32[getMemAddr(i, j, numDocs, embDim)] = emb2D[i][j];
}

void copyAsBF16(const vector<vector<float>> &emb2D, __nv_bfloat16 *d_bf16, int numDocs, int embDim)
{
    for (int i = 0; i < numDocs; i++)
        for (int j = 0; j < embDim; j++)
            d_bf16[getMemAddr(i, j, numDocs, embDim)] = (__nv_bfloat16)emb2D[i][j];
}

void copyAsBF162(const vector<vector<float>> &emb2D, __nv_bfloat162 *d_bf162, int numDocs, int embDim2)
{
    for (int i = 0; i < numDocs; i++)
    {
        for (int j2 = 0; j2 < embDim2; j2++)
        {
            int j = j2 * 2;
            float float1 = emb2D[i][j];
            float float2 = emb2D[i][j+1];
            d_bf162[getMemAddr(i, j2, numDocs, embDim2)] = __floats2bfloat162_rn(float1, float2);
        }
    }
}

void copyAsFloat4(const vector<vector<float>> &emb2D, float4 *d_float4, int numDocs, int embDim4)
{
    for (int i = 0; i < numDocs; i++)
    {
        for (int j4 = 0; j4 < embDim4; j4++)
        {
            int j = j4 * 4;
            d_float4[getMemAddr(i, j4, numDocs, embDim4)].x = emb2D[i][j];
            d_float4[getMemAddr(i, j4, numDocs, embDim4)].y = emb2D[i][j+1];
            d_float4[getMemAddr(i, j4, numDocs, embDim4)].z = emb2D[i][j+2];
            d_float4[getMemAddr(i, j4, numDocs, embDim4)].w = emb2D[i][j+3];
        }
    }
}

vector<Doc> referenceAlgo(const vector<vector<float>> &docEmb2D, const vector<vector<float>> &reqEmb2D, const vector<Doc> &v_doc)
{
    int numDocs = docEmb2D.size();
    int embDim = docEmb2D[0].size();
    vector<Doc> v_rst(v_doc.size());
    for (int d = 0; d < v_doc.size(); d++)
    {
        Doc doc = v_doc[d];
        int i = doc.docIdx;
        double acc = 0;
        for (int j = 0; j < embDim; j++)
            acc += docEmb2D[i][j] * reqEmb2D[0][j];
        doc.score = acc;
        v_rst[d] = doc;
    }
    return v_rst;
}

vector<Doc> genRandActiveDocs(int numDocs, int numActiveDocs)
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
    sort(v_doc.begin(), v_doc.end(), [](const Doc &a, const Doc &b) { return a.docIdx < b.docIdx; });
    for (int i = 0; i < numActiveDocs-1; i++)
        assert(v_doc[i].docIdx < v_doc[i+1].docIdx);
    return v_doc;
}

void runExp(int numDocs, int embDim, float density)
{
    cout << "Running experiment with numDocs=" << numDocs << ", embDim=" << embDim << ", density=" << density << endl;

    int numActiveDocs = (int)(numDocs * density);
    int embDim2 = embDim / 2;
    int embDim4 = embDim / 4;
    assert(embDim2 * 2 == embDim);
    assert(embDim4 * 4 == embDim);

    vector<vector<float>> docEmb2D = genRandEmb(numDocs, embDim);
    vector<vector<float>> reqEmb2D = genRandEmb(1, embDim);
    
    vector<Doc> v_doc = genRandActiveDocs(numDocs, numActiveDocs);
    Doc *d_doc = nullptr;
    CHECK_CUDA(cudaMallocManaged(&d_doc, numActiveDocs * sizeof(Doc)));
    CHECK_CUDA(cudaMemcpy(d_doc, v_doc.data(), numActiveDocs * sizeof(Doc), cudaMemcpyHostToDevice));

    vector<Doc> v_doc_ref = referenceAlgo(docEmb2D, reqEmb2D, v_doc);

    float *d_docEmb_fp32 = nullptr;
    float *d_reqEmb_fp32 = nullptr;
    CHECK_CUDA(cudaMallocManaged(&d_docEmb_fp32, numDocs * embDim * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&d_reqEmb_fp32, embDim * sizeof(float)));
    copyAsFP32(docEmb2D, d_docEmb_fp32, numDocs, embDim);
    copyAsFP32(reqEmb2D, d_reqEmb_fp32, 1, embDim);

    __nv_bfloat16 *d_docEmb_bf16 = nullptr;
    __nv_bfloat16 *d_reqEmb_bf16 = nullptr;
    CHECK_CUDA(cudaMallocManaged(&d_docEmb_bf16, numDocs * embDim * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMallocManaged(&d_reqEmb_bf16, embDim * sizeof(__nv_bfloat16)));
    copyAsBF16(docEmb2D, d_docEmb_bf16, numDocs, embDim);
    copyAsBF16(reqEmb2D, d_reqEmb_bf16, 1, embDim);

    __nv_bfloat162 *d_docEmb_bf162 = nullptr;
    __nv_bfloat162 *d_reqEmb_bf162 = nullptr;
    CHECK_CUDA(cudaMallocManaged(&d_docEmb_bf162, numDocs * embDim2 * sizeof(__nv_bfloat162)));
    CHECK_CUDA(cudaMallocManaged(&d_reqEmb_bf162, embDim2 * sizeof(__nv_bfloat162)));
    copyAsBF162(docEmb2D, d_docEmb_bf162, numDocs, embDim2);
    copyAsBF162(reqEmb2D, d_reqEmb_bf162, 1, embDim2);

    float4 *d_docEmb_float4 = nullptr;
    float4 *d_reqEmb_float4 = nullptr;
    CHECK_CUDA(cudaMallocManaged(&d_docEmb_float4, numDocs * embDim4 * sizeof(float4)));
    CHECK_CUDA(cudaMallocManaged(&d_reqEmb_float4, embDim4 * sizeof(float4)));
    copyAsFloat4(docEmb2D, d_docEmb_float4, numDocs, embDim4);
    copyAsFloat4(reqEmb2D, d_reqEmb_float4, 1, embDim2);
    cudaDeviceSynchronize();

    float timeMs, rmse;

    cout << fixed << setprecision(10);
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
    rmse = computeRMSE(v_doc_ref, v_doc_fp32_fp64);
    cout << "time = " << timeMs << " ms" << ", rmse = " << rmse << endl;

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
    rmse = computeRMSE(v_doc_ref, v_doc_fp32_fp32);
    cout << "time = " << timeMs << " ms" << ", rmse = " << rmse << endl;

    cout << "data type = float4, accumulator type = fp64, ";
    timeMs = 0;
    for (int t = -3; t < kNumTrials; t++)
    {
        float timeMs1 = score_float4<double>(d_docEmb_float4, d_reqEmb_float4, d_doc, numDocs, numActiveDocs, embDim4);
        if (t >= 0)
            timeMs += timeMs1;
    }
    timeMs /= kNumTrials;
    vector<Doc> v_doc_float4_fp64(numActiveDocs);
    CHECK_CUDA(cudaMemcpy(v_doc_float4_fp64.data(), d_doc, numActiveDocs * sizeof(Doc), cudaMemcpyDeviceToHost));
    rmse = computeRMSE(v_doc_ref, v_doc_float4_fp64);
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
    rmse = computeRMSE(v_doc_ref, v_doc_bf16_fp64);
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
    rmse = computeRMSE(v_doc_ref, v_doc_bf16_fp32);
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
    rmse = computeRMSE(v_doc_ref, v_doc_bf16_bf16);
    cout << "time = " << timeMs << " ms" << ", rmse = " << rmse << endl;

    cout << "data type = bf162, accumulator type = fp64, ";
    timeMs = 0;
    for (int t = -3; t < kNumTrials; t++)
    {
        float timeMs1 = score_bf162<double>(d_docEmb_bf162, d_reqEmb_bf162, d_doc, numDocs, numActiveDocs, embDim2);
        if (t >= 0)
            timeMs += timeMs1;
    }
    timeMs /= kNumTrials;
    vector<Doc> v_doc_bf162_fp64(numActiveDocs);
    CHECK_CUDA(cudaMemcpy(v_doc_bf162_fp64.data(), d_doc, numActiveDocs * sizeof(Doc), cudaMemcpyDeviceToHost));
    rmse = computeRMSE(v_doc_ref, v_doc_bf162_fp64);
    cout << "time = " << timeMs << " ms" << ", rmse = " << rmse << endl;

    cout << "data type = bf162, accumulator type = fp32, ";
    timeMs = 0;
    for (int t = -3; t < kNumTrials; t++)
    {
        float timeMs1 = score_bf162<float>(d_docEmb_bf162, d_reqEmb_bf162, d_doc, numDocs, numActiveDocs, embDim2);
        if (t >= 0)
            timeMs += timeMs1;
    }
    timeMs /= kNumTrials;
    vector<Doc> v_doc_bf162_fp32(numActiveDocs);
    CHECK_CUDA(cudaMemcpy(v_doc_bf162_fp32.data(), d_doc, numActiveDocs * sizeof(Doc), cudaMemcpyDeviceToHost));
    rmse = computeRMSE(v_doc_ref, v_doc_bf162_fp32);
    cout << "time = " << timeMs << " ms" << ", rmse = " << rmse << endl;

    cout << "data type = bf162, accumulator type = bf16, ";
    timeMs = 0;
    for (int t = -3; t < kNumTrials; t++)
    {
        float timeMs1 = score_bf162<__nv_bfloat16>(d_docEmb_bf162, d_reqEmb_bf162, d_doc, numDocs, numActiveDocs, embDim2);
        if (t >= 0)
            timeMs += timeMs1;
    }
    timeMs /= kNumTrials;
    vector<Doc> v_doc_bf162_bf16(numActiveDocs);
    CHECK_CUDA(cudaMemcpy(v_doc_bf162_bf16.data(), d_doc, numActiveDocs * sizeof(Doc), cudaMemcpyDeviceToHost));
    rmse = computeRMSE(v_doc_ref, v_doc_bf162_bf16);
    cout << "time = " << timeMs << " ms" << ", rmse = " << rmse << endl;
}

int main()
{
    runExp(1000000, 1024, 0.5);

    return 0;
}