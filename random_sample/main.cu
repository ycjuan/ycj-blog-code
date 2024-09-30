#include <random>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <unordered_set>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include "util.cuh"

using namespace std;

const int kNumTrials = 10;
const int kBlockSize = 512;
const vector<int> kNumDocsPerPartition = {100, 1000, 10000, 100000, 1000000, 10000000};
const int kNumPartitions = kNumDocsPerPartition.size();
const int kMaxNumDocs = kNumDocsPerPartition[kNumPartitions - 1];
const int kNumDocsTotal = accumulate(kNumDocsPerPartition.begin(), kNumDocsPerPartition.end(), 0);
const int kSampleSize = 250;

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
    int partitionIdx; // for example, a partition can be things like "country", "language", etc.
};

void checkSample(Doc *d_docDst, int numSampled)
{
    cout << "============" << endl;
    cout << "numSampled: " << numSampled << endl;
    vector<vector<Doc>> sample(kNumPartitions);
    for (int i = 0; i < numSampled; i++)
    {
        Doc doc = d_docDst[i];
        sample[doc.partitionIdx].push_back(doc);
    }
    for (int partitionIdx = 0; partitionIdx < kNumPartitions; partitionIdx++)
        cout << "partitionIdx: " << partitionIdx << ", numSampled: " << sample[partitionIdx].size() << endl;
    cout << "============" << endl;
}

namespace classicRandomSample
{
    __global__ void kernel(Doc *d_docBuffer, int *d_sampleIdxBuffer, Doc *d_docDst, int kSampleSize)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < kSampleSize)
        {
            d_docDst[i] = d_docBuffer[d_sampleIdxBuffer[i]];
        }
    }

    struct Predicator
    {
        int partitionIdx;
        Predicator(int partitionIdx) : partitionIdx(partitionIdx) {}

        __host__ __device__ bool operator()(const Doc x)
        {
            return x.partitionIdx == partitionIdx;
        }
    };

    int sample(Doc *d_docSrc, Doc *d_docDst, Doc *d_docBuffer, int *d_sampleIdxBuffer)
    {
        int sampleSizeAgg = 0;
        for (int partitionIdx = 0; partitionIdx < kNumPartitions; partitionIdx++)
        {
            // extract documents of this partition
            int kNumDocs = kNumDocsPerPartition[partitionIdx];
            Doc *d_docBufferEndPtr = thrust::copy_if(thrust::device, d_docSrc, d_docSrc + kNumDocsTotal, d_docBuffer, Predicator(partitionIdx));
            if (d_docBufferEndPtr - d_docBuffer != kNumDocs)
                throw runtime_error("Error: d_docBufferEndPtr - d_docBuffer != kNumDocs");

            // if num docs in this partition is less than kSampleSize, just copy all
            if (kSampleSize >= kNumDocs)
            {
                CHECK_CUDA(cudaMemcpy(d_docDst + sampleSizeAgg, d_docBuffer, kNumDocs * sizeof(Doc), cudaMemcpyDeviceToDevice))
                sampleSizeAgg += kNumDocs;
                continue;
            }

            // generate random indexes
            default_random_engine generator;
            uniform_int_distribution<int> distribution(0, kNumDocs);
            unordered_set<int> sampledIdxSet;
            while (sampledIdxSet.size() < kSampleSize)
            {
                int idx = distribution(generator);
                sampledIdxSet.insert(idx);
            }

            // do sampling
            int gridSize = (int)ceil((double)(kSampleSize + 1) / kBlockSize);
            kernel<<<gridSize, kBlockSize>>>(d_docBuffer, d_sampleIdxBuffer, d_docDst + sampleSizeAgg, kSampleSize);
            cudaDeviceSynchronize();
            CHECK_CUDA(cudaGetLastError());
            sampleSizeAgg += kSampleSize;
        }
        return sampleSizeAgg;
    }

    void runExp(Doc *d_docSrc, Doc *d_docDst)
    {
        Doc *d_docBuffer = nullptr;
        int *d_sampleIdxBuffer = nullptr;
        CHECK_CUDA(cudaMalloc(&d_docBuffer, kMaxNumDocs * sizeof(Doc)));
        CHECK_CUDA(cudaMalloc(&d_sampleIdxBuffer, kSampleSize * sizeof(int)));

        double timeMs = 0;
        for (int t = -3; t < kNumTrials; t++)
        {
            CudaTimer timer;
            timer.tic();
            int sampleSizeAgg = sample(d_docSrc, d_docDst, d_docBuffer, d_sampleIdxBuffer);
            if (t >= 0)
                timeMs += timer.tocMs();
            if (t == 0)
                checkSample(d_docDst, sampleSizeAgg);
        }
        timeMs /= kNumTrials;
        cout << "[classicRandomSample] timeMs: " << timeMs << " ms" << endl;

        cudaFree(d_docBuffer);
        cudaFree(d_sampleIdxBuffer);
    }
}

namespace adhocRandomSampleGreedy
{
    __managed__ int docDstCurrIdx;

    struct KernelParam
    {
        Doc *d_docSrc;
        Doc *d_docDst;
        int *d_partitionCounter;
        int numDocsTotal;
        int randomStartingPoint;
        int sampleSize;
    };

    __global__ void kernel(KernelParam param)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= param.numDocsTotal)
            return;
        
        int docIdx = (param.randomStartingPoint + i) % param.numDocsTotal;
        int partitionIdx = param.d_docSrc[docIdx].partitionIdx;

        if (param.d_partitionCounter[partitionIdx] >= param.sampleSize)
            return;
        int partitionCountOld = atomicAdd(param.d_partitionCounter + partitionIdx, 1);
        if (partitionCountOld >= param.sampleSize)
        {
            atomicSub(param.d_partitionCounter + partitionIdx, 1);
            return;
        }
        int currIdxOld = atomicAdd(&docDstCurrIdx, 1);
        param.d_docDst[currIdxOld] = param.d_docSrc[docIdx];
    }

    int sample(Doc *d_docSrc, Doc *d_docDst, int *d_partitionCounter)
    {
        CHECK_CUDA(cudaMemset(d_partitionCounter, 0, kNumPartitions * sizeof(int)))
        default_random_engine generator;
        uniform_int_distribution<int> distribution(0, kNumDocsTotal);
        int randomStartingPoint = distribution(generator);

        docDstCurrIdx = 0;
        int gridSize = (int)ceil((double)(kNumDocsTotal + 1) / kBlockSize);
        KernelParam param;
        param.d_docSrc = d_docSrc;
        param.d_docDst = d_docDst;
        param.d_partitionCounter = d_partitionCounter;
        param.randomStartingPoint = randomStartingPoint;
        param.numDocsTotal = kNumDocsTotal;
        param.sampleSize = kSampleSize;
        kernel<<<gridSize, kBlockSize>>>(param);
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());

        int numSampled = 0;
        for (int partitionIdx = 0; partitionIdx < kNumPartitions; partitionIdx++)
            numSampled += d_partitionCounter[partitionIdx];
        return numSampled;
    }

    void runExp(Doc *d_docSrc, Doc *d_docDst)
    {
        int *d_partitionCounter = nullptr;
        CHECK_CUDA(cudaMallocManaged(&d_partitionCounter, kSampleSize * sizeof(int)));

        double timeMs = 0;
        for (int t = -3; t < kNumTrials; t++)
        {
            CudaTimer timer;
            timer.tic();
            int sampleSizeAgg = sample(d_docSrc, d_docDst, d_partitionCounter);
            if (t >= 0)
                timeMs += timer.tocMs();
            if (t == 0)
                checkSample(d_docDst, sampleSizeAgg);
        }
        timeMs /= kNumTrials;
        cout << "[adhocRandomSampleGreedy] timeMs: " << timeMs << " ms" << endl;

        cudaFree(d_partitionCounter);
    }
}


int main()
{
    cout << "kNumDocsPerPartition: ";
    for (auto kNumDocs : kNumDocsPerPartition)
    {
        cout << kNumDocs << " ";
    }
    cout << endl;
    cout << "kNumDocsTotal: " << kNumDocsTotal << endl;
    cout << "kSampleSize: " << kSampleSize << endl;
    cout << "kNumTrials: " << kNumTrials << endl;
    cout << "kMaxNumDocs: " << kMaxNumDocs << endl;
    cout << "kNumPartitions: " << kNumPartitions << endl;

    Doc *d_docSrc = nullptr;
    Doc *d_docDst = nullptr;
    CHECK_CUDA(cudaMallocManaged(&d_docSrc, kNumDocsTotal * sizeof(Doc)));
    CHECK_CUDA(cudaMallocManaged(&d_docDst, kNumPartitions * kSampleSize * sizeof(Doc)));

    int docIdx = 0;
    for (int partitionIdx = 0; partitionIdx < kNumPartitions; partitionIdx++)
    {
        int kNumDocs = kNumDocsPerPartition[partitionIdx];
        for (int i = 0; i < kNumDocs; i++)
        {
            d_docSrc[docIdx].docIdx = docIdx;
            d_docSrc[docIdx].partitionIdx = partitionIdx;
            docIdx++;
        }
    }
    if (docIdx != kNumDocsTotal)
        throw runtime_error("Error: docIdx != kNumDocsTotal");

    classicRandomSample::runExp(d_docSrc, d_docDst);
    adhocRandomSampleGreedy::runExp(d_docSrc, d_docDst);

    cudaFree(d_docSrc);
    cudaFree(d_docDst);

    return 0;
}
