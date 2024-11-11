#include <random>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <algorithm>

#include "util.cuh"

const int kNumDocs = 1000000;
const int kEmbDim = 128;
const int kNumTrials = 10;
const int kMetaSize = 2; // DON'T CHANGE THIS VALUE
const int kBlockSize = 256;
const int kGetSetCount = 200;

using namespace std;

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "[main.cu] CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

struct Meta
{
    int idxBegin;
    int idxEnd;
    float weight;
};

__global__ void kernel(float *d_doc, float *d_req, float *d_rst, Meta *d_meta, int numDocs, int embDim, int metaSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numDocs)
    {
        double rst = 0;
        for (int j = 0; j < metaSize; j++)
        {
            int idxBegin = d_meta[j].idxBegin;
            int idxEnd = d_meta[j].idxEnd;
            float weight = d_meta[j].weight;
            for (int k = idxBegin; k < idxEnd; k++)
                rst += d_doc[k * kNumDocs + i] * d_req[k] * weight;
        }
        d_rst[i] = rst;
    }
}

vector<float> runGpu(float *d_doc, float *d_req, float *d_rst, Meta *d_meta)
{
    int gridSize = (int)ceil((double)(kNumDocs + 1) / kBlockSize);
    double timeMs = 0;
    for (int t = -3; t < kNumTrials; t++)
    {
        CudaTimer timer;
        timer.tic();
        kernel<<<gridSize, kBlockSize>>>(d_doc, d_req, d_rst, d_meta, kNumDocs, kEmbDim, kMetaSize);
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());
        if (t >= 0)
            timeMs += timer.tocMs();
    }
    timeMs /= kNumTrials;
    cout << "timeMs: " << timeMs << " ms" << endl;

    vector<float> v_rst(kNumDocs);
    CHECK_CUDA(cudaMemcpy(v_rst.data(), d_rst, kNumDocs * sizeof(float), cudaMemcpyDeviceToHost));
    return v_rst;
}

vector<float> runCpu(const vector<float> &v_docEmb, const vector<float> &v_reqEmb, const vector<Meta> &v_meta)
{
    vector<float> v_rst(kNumDocs);
    for (int i = 0; i < kNumDocs; i++)
    {
        double rst = 0;
        for (int j = 0; j < kMetaSize; j++)
        {
            int idxBegin = v_meta[j].idxBegin;
            int idxEnd = v_meta[j].idxEnd;
            float weight = v_meta[j].weight;
            for (int k = idxBegin; k < idxEnd; k++)
                rst += v_docEmb[k * kNumDocs + i] * v_reqEmb[k] * weight;
        }
        v_rst[i] = rst;
    }
    return v_rst;
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

void testGetSetValue(float *d_rst, bool doSet)
{   
    default_random_engine generator;
    uniform_real_distribution<float> distribution(-1.0, 1.0);
    
    vector<int> v_idx(kNumDocs);
    for (int i = 0; i < kNumDocs; i++)
        v_idx[i] = i;
    shuffle(v_idx.begin(), v_idx.end(), generator);
    assert(v_idx[0] != 0);
    assert(v_idx[kNumDocs] != kNumDocs - 1);
    v_idx.resize(kGetSetCount);

    vector<float> v_buffer(kNumDocs);
    for (int i = 0; i < kNumDocs; i++)
        v_buffer[i] = distribution(generator);

    CudaTimer timer;
    timer.tic();
    for (auto idx : v_idx)
    {
        if (doSet)
            d_rst[idx] = v_buffer[idx];
        else
            v_buffer[idx] = d_rst[idx];
    }
    float timeMs = (double)timer.tocMs() / kGetSetCount;
    cout << "doSet = " << doSet << ", timeMs: " << fixed << timeMs << " ms" << endl;
}

int main()
{
    cout << "kNumDocs: " << kNumDocs << ", kEmbDim: " << kEmbDim << ", kMetaSize: " << kMetaSize << ", kGetSetCount: " << kGetSetCount << endl;
    // Some tools
    default_random_engine generator;
    uniform_real_distribution<float> distribution(-1.0, 1.0);

    // Generate random data
    vector<float> v_docEmb(kNumDocs * kEmbDim);
    for (auto &v : v_docEmb)
        v = distribution(generator);

    vector<float> v_reqEmb(kEmbDim);
    for (auto &v : v_reqEmb)
        v = distribution(generator);

    Meta meta1;
    meta1.idxBegin = 0;
    meta1.idxEnd = 64;
    meta1.weight = 1.1;

    Meta meta2;
    meta2.idxBegin = 64;
    meta2.idxEnd = 128;
    meta2.weight = 2.2;

    vector<Meta> v_meta = {meta1, meta2};
    
    // Normal memory
    float *d_docEmb = nullptr;
    float *d_reqEmb = nullptr;
    float *d_rst = nullptr;
    Meta *d_meta = nullptr;
    CHECK_CUDA(cudaMalloc(&d_docEmb, kNumDocs * kEmbDim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_reqEmb, kEmbDim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_rst, kNumDocs * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_meta, kMetaSize * sizeof(Meta)));
    CHECK_CUDA(cudaMemcpy(d_docEmb, v_docEmb.data(), kNumDocs * kEmbDim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_reqEmb, v_reqEmb.data(), kEmbDim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_meta, v_meta.data(), kMetaSize * sizeof(Meta), cudaMemcpyHostToDevice));

    // Managed memory
    float *d_docEmb_managed = nullptr;
    float *d_reqEmb_managed = nullptr;
    float *d_rst_managed = nullptr;
    Meta *d_meta_managed = nullptr;

    // The code below demostrates that cudaMallocManaged may use CPU memory
    cout << "memroy usage (before cudaMallocManaged): " << getCpuRamUsageMiB() << " MiB" << endl;
    CHECK_CUDA(cudaMallocManaged(&d_docEmb_managed, kNumDocs * kEmbDim * sizeof(float)));
    cout << "memroy usage (after cudaMallocManaged): " << getCpuRamUsageMiB() << " MiB" << endl;
    for (int i = 0; i < kNumDocs * kEmbDim; i++)
        d_docEmb_managed[i] = 0;
    cout << "memroy usage (after writing to d_docEmb_managed using for loop): " << getCpuRamUsageMiB() << " MiB" << endl;
    
    CHECK_CUDA(cudaMallocManaged(&d_reqEmb_managed, kEmbDim * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&d_rst_managed, kNumDocs * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&d_meta_managed, kMetaSize * sizeof(Meta)));
    CHECK_CUDA(cudaMemcpy(d_docEmb_managed, v_docEmb.data(), kNumDocs * kEmbDim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_reqEmb_managed, v_reqEmb.data(), kEmbDim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_meta_managed, v_meta.data(), kMetaSize * sizeof(Meta), cudaMemcpyHostToDevice));

    vector<float> v_rst0 = runCpu(v_docEmb, v_reqEmb, v_meta);

    cout << "d_docEmb, d_reqEmb, d_rst, d_meta" << endl;
    vector<float> v_rst1 = runGpu(d_docEmb, d_reqEmb, d_rst, d_meta);
    assert(compareRst(v_rst0, v_rst1));

    cout << "d_docEmb_managed, d_reqEmb, d_rst, d_meta" << endl;
    vector<float> v_rst2 = runGpu(d_docEmb_managed, d_reqEmb, d_rst, d_meta);
    assert(compareRst(v_rst0, v_rst2));

    cout << "d_docEmb_managed, d_reqEmb, d_rst_managed, d_meta" << endl;
    vector<float> v_rst3 = runGpu(d_docEmb_managed, d_reqEmb, d_rst_managed, d_meta);
    assert(compareRst(v_rst0, v_rst3));
    assert(v_rst3[0] == d_rst_managed[0]); // This works. We can directly access GPU array
    assert(v_rst3[99] == d_rst_managed[99]);
    assert(v_rst3[kNumDocs-1] == d_rst_managed[kNumDocs-1]);

    cout << "d_docEmb, d_reqEmb, d_rst, d_meta_managed" << endl;
    vector<float> v_rst4 = runGpu(d_docEmb, d_reqEmb, d_rst, d_meta_managed);
    assert(compareRst(v_rst0, v_rst4));
    // assert(v_rst4[0] == d_rst[0]); // This leads to segmentation fault

    // Alter the weight
    v_meta[0].weight = 1.2;
    v_meta[1].weight = 2.4;
    //d_meta[0].weight = v_meta[0].weight; // This leads to segmentation fault
    //d_meta[1].weight = v_meta[1].weight;
    d_meta_managed[0].weight = v_meta[0].weight; // This is fine
    d_meta_managed[1].weight = v_meta[1].weight;

    vector<float> v_rst_b0 = runCpu(v_docEmb, v_reqEmb, v_meta);
    vector<float> v_rst_b1 = runGpu(d_docEmb, d_reqEmb, d_rst, d_meta_managed);
    assert(compareRst(v_rst_b0, v_rst_b1));

    testGetSetValue(d_rst_managed, false);
    testGetSetValue(d_rst_managed, true);

    // Should free memory; please forgive my laziness...
    return 0;
}