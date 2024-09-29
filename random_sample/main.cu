#include <random>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <algorithm>

#include "util.cuh"

const int kNumDocs = 1000000;
const int kNumTrials = 10;
const int kBlockSize = 512;
const float kSampleRate = 0.1;

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

struct Doc
{
    int docIdx;
    bool isSelected;
};

__global__ void sampleKernelSimpleRandCopyIf(Doc *d_docSrc, Doc *d_docDst, int numDocs, int seed, int invSampleRate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numDocs)
    {
        Doc &doc = d_docSrc[i];
        int randNum = doc.docIdx + seed;
        doc.isSelected = (randNum % invSampleRate == 0);
    }
}

void sampleFuncSimpleRandCopyIf(Doc *d_docSrc, Doc *d_docDst, int numDocs, float sampleRate)
{
    int gridSize = (int)ceil((double)(kNumDocs + 1) / kBlockSize);
    int invSampleRate = 1.0 / sampleRate;
    double timeMs = 0;
    for (int t = -3; t < kNumTrials; t++)
    {
        CudaTimer timer;
        timer.tic();
        sampleKernelSimpleRandCopyIf<<<gridSize, kBlockSize>>>(d_docSrc, d_docDst, numDocs, invSampleRate, t);
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());
        if (t >= 0)
            timeMs += timer.tocMs();
    }
    timeMs /= kNumTrials;
    cout << "[sampleFuncSimpleRandCopyIf] timeMs: " << timeMs << " ms" << endl;
}

//chunkRandom

int main()
{
    cout << "kNumDocs: " << kNumDocs << ", kSampleRate: " << kSampleRate << endl;

    Doc *d_docSrc = nullptr;
    Doc *d_docDst = nullptr;
    CHECK_CUDA(cudaMallocManaged(&d_docSrc, kNumDocs * sizeof(Doc)));
    CHECK_CUDA(cudaMallocManaged(&d_docDst, kNumDocs * sizeof(Doc)));

    for (int i = 0; i < kNumDocs; i++)
        d_docSrc[i].docIdx = i;

    sampleFuncSimpleRandCopyIf(d_docSrc, d_docDst, kNumDocs, kSampleRate);

    cudaFree(d_docSrc);
    cudaFree(d_docDst);

    return 0;
}