#include <random>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

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

namespace pseudoRandomCopyIf
{
    __global__ void kernel(Doc *d_docSrc, Doc *d_docDst, int numDocs, int seed, int invSampleRate)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < numDocs)
        {
            Doc &doc = d_docSrc[i];
            int randNum = doc.docIdx + seed;
            doc.isSelected = (randNum % invSampleRate == 0);
        }
    }

    struct Predicator
    {
        __host__ __device__ bool operator()(const Doc x)
        {
            return x.isSelected;
        }
    };

    void sample(Doc *d_docSrc, Doc *d_docDst, int numDocs, float sampleRate)
    {
        int gridSize = (int)ceil((double)(kNumDocs + 1) / kBlockSize);
        int invSampleRate = 1.0 / sampleRate;
        double timeMs = 0;
        for (int t = -3; t < kNumTrials; t++)
        {
            CudaTimer timer;
            timer.tic();
            kernel<<<gridSize, kBlockSize>>>(d_docSrc, d_docDst, numDocs, t, invSampleRate);
            cudaDeviceSynchronize();
            CHECK_CUDA(cudaGetLastError());
            Doc *d_endPtr = thrust::copy_if(thrust::device, d_docSrc, d_docSrc + numDocs, d_docDst, Predicator());
            cudaDeviceSynchronize();
            CHECK_CUDA(cudaGetLastError());
            int numSampled = d_endPtr - d_docDst;
            if (t >= 0)
                timeMs += timer.tocMs();
            cout << "[pseudoRandCopyIf] " << "t: " << t << "numSampled: " << numSampled << endl;
        }
        timeMs /= kNumTrials;
        cout << "[pseudoRandCopyIf] timeMs: " << timeMs << " ms" << endl;
    }
}

namespace pseudoRandomAtomicAdd
{
    __managed__ int currIdx = 0;

    __global__ void kernel(Doc *d_docSrc, Doc *d_docDst, int numDocs, int seed, int invSampleRate)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < numDocs)
        {
            Doc &doc = d_docSrc[i];
            int randNum = doc.docIdx + seed;
            doc.isSelected = (randNum % invSampleRate == 0);
            if (doc.isSelected)
            {
                int idx = atomicAdd(&currIdx, 1);
                d_docDst[idx] = doc;
            }
        }
    }

    void sample(Doc *d_docSrc, Doc *d_docDst, int numDocs, float sampleRate)
    {
        int gridSize = (int)ceil((double)(kNumDocs + 1) / kBlockSize);
        int invSampleRate = 1.0 / sampleRate;
        double timeMs = 0;
        for (int t = -3; t < kNumTrials; t++)
        {
            CudaTimer timer;
            timer.tic();
            currIdx = 0;
            kernel<<<gridSize, kBlockSize>>>(d_docSrc, d_docDst, numDocs, t, invSampleRate);
            cudaDeviceSynchronize();
            CHECK_CUDA(cudaGetLastError());
            if (t >= 0)
                timeMs += timer.tocMs();
            cout << "[pseudoRandAtomicAdd] " << "t: " << t << "numSampled: " << currIdx << endl;
        }
        timeMs /= kNumTrials;
        cout << "[pseudoRandCopyIf] timeMs: " << timeMs << " ms" << endl;
    }
}

namespace randomChunk
{
    void sample(Doc *d_docSrc, Doc *d_docDst, int numDocs, float sampleRate)
    {
        int sampleSize = numDocs * sampleRate;

        default_random_engine generator;
        uniform_int_distribution<int> distribution(0, numDocs - sampleSize);
        double timeMs = 0;
        for (int t = -3; t < kNumTrials; t++)
        {
            CudaTimer timer;
            timer.tic();
            int indexBegin = distribution(generator);
            cudaMemcpy(d_docDst + indexBegin, d_docSrc, sampleSize * sizeof(Doc), cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
            CHECK_CUDA(cudaGetLastError());
            cout << "[randomChunk] " << "t: " << t << "numSampled: " << sampleSize << endl;
            if (t >= 0)
                timeMs += timer.tocMs();
        }
        timeMs /= kNumTrials;
        cout << "[randomChunk] timeMs: " << timeMs << " ms" << endl;
    }
}


int main()
{
    cout << "kNumDocs: " << kNumDocs << ", kSampleRate: " << kSampleRate << endl;

    Doc *d_docSrc = nullptr;
    Doc *d_docDst = nullptr;
    CHECK_CUDA(cudaMallocManaged(&d_docSrc, kNumDocs * sizeof(Doc)));
    CHECK_CUDA(cudaMallocManaged(&d_docDst, kNumDocs * sizeof(Doc)));

    for (int i = 0; i < kNumDocs; i++)
        d_docSrc[i].docIdx = i;

    pseudoRandomCopyIf::sample(d_docSrc, d_docDst, kNumDocs, kSampleRate);
    pseudoRandomAtomicAdd::sample(d_docSrc, d_docDst, kNumDocs, kSampleRate);
    randomChunk::sample(d_docSrc, d_docDst, kNumDocs, kSampleRate);

    cudaFree(d_docSrc);
    cudaFree(d_docDst);

    return 0;
}