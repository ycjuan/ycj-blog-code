#include <string>
#include <stdexcept>
#include <iostream>
#include <random>
#include <sstream>
#include <cublas_v2.h>
#include <type_traits>
#include <algorithm>

#include "util.cuh"
#include "common.cuh"
#include "methods.cuh"

using namespace std;

int kNumDocs = 1 << 10;
int kNumReqs = 1 << 2;
int kEmbDim = 1 << 4;
int kNumTrials = 3;
double kDocDensity = 0.1;
MemLayout kMemLayoutDoc = COL_MAJOR;
MemLayout kMemLayoutReq = ROW_MAJOR;
MemLayout kMemLayoutRstCpu = COL_MAJOR;
MemLayout kMemLayoutRstGpuCuda = COL_MAJOR;
MemLayout kMemLayoutRstGpuCublas = COL_MAJOR;

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

Data genData()
{
    Data data;
    data.numDocs = kNumDocs;
    data.numReqs = kNumReqs;
    data.embDim = kEmbDim;
    data.docMemLayout = kMemLayoutDoc;
    data.reqMemLayout = kMemLayoutReq;
    data.rstLayoutCpu = kMemLayoutRstCpu;
    data.rstLayoutGpuKernel = kMemLayoutRstGpuCuda;
    data.rstLayoutGpuCublas = kMemLayoutRstGpuCublas;
    data.print();
    
    CHECK_CUDA(cudaMallocManaged(&data.d_doc, (size_t)data.numDocs * data.embDim * sizeof(T)));
    CHECK_CUDA(cudaMallocManaged(&data.d_req, (size_t)data.numReqs * data.embDim * sizeof(T)));
    CHECK_CUDA(cudaMallocManaged(&data.d_PairsToScore, (size_t)data.numDocs * data.numReqs * sizeof(Pair)));
    CHECK_CUDA(cudaMallocManaged(&data.d_rstCuda, (size_t)data.numDocs * data.numReqs * sizeof(Pair)));
    CHECK_CUDA(cudaMallocManaged(&data.d_rstCusparse, (size_t)data.numDocs * data.numReqs * sizeof(Pair)));
    CHECK_CUDA(cudaMallocHost(&data.h_rstCpu, (size_t)data.numDocs * data.numReqs * sizeof(Pair)));

    default_random_engine generator;
    uniform_real_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < data.numDocs * data.embDim; i++)
        data.d_doc[i] = (T)distribution(generator);
    for (int i = 0; i < data.numReqs * data.embDim; i++)
        data.d_req[i] = (T)distribution(generator);
    
    data.numPairsToScore = 0;
    for (int reqIdx = 0; reqIdx < data.numReqs; reqIdx++)
    {
        int numEligibleDocsPerReq = int(kDocDensity * data.numDocs);
        vector<int> v_docIdx1D(data.numDocs);
        for (int i = 0; i < data.numDocs; i++)
            v_docIdx1D[i] = i;
        shuffle(v_docIdx1D.begin(), v_docIdx1D.end(), generator);
        sort(v_docIdx1D.begin(), v_docIdx1D.begin() + numEligibleDocsPerReq);
        cout << "first 10 eligible doc indices for req " << reqIdx << ": ";
        for (int i = 0; i < 10; i++)
            cout << v_docIdx1D[i] << " ";
        cout << endl;
        for (int docIdx = 0; docIdx < numEligibleDocsPerReq; docIdx++)
        {
            Pair pair;
            pair.reqIdx = reqIdx;
            pair.docIdx = v_docIdx1D[docIdx];
            data.d_PairsToScore[data.numPairsToScore++] = pair;
        }
    }
    
    return data;
}

void checkData(Data data)
{
    sort(data.h_rstCpu, data.h_rstCpu + data.numPairsToScore, pairComparatorDocFirst);
    sort(data.d_rstCuda, data.d_rstCuda + data.numPairsToScore, pairComparatorDocFirst);
    sort(data.d_rstCusparse, data.d_rstCusparse + data.numPairsToScore, pairComparatorDocFirst);
    for (size_t pairIdx = 0; pairIdx < data.numPairsToScore; pairIdx++)
    {
        Pair pairCpu = data.h_rstCpu[pairIdx];
        Pair pairCuda = data.d_rstCuda[pairIdx];
        Pair pairCusparse = data.d_rstCusparse[pairIdx];

        if (pairCpu.reqIdx != pairCuda.reqIdx ||
            pairCpu.docIdx != pairCuda.docIdx ||
            abs(pairCpu.score - pairCuda.score) > 1e-3)
        {
            cout << "Mismatch at pairIdx " << pairIdx << endl;
            cout << "CPU: " << pairCpu.reqIdx << " " << pairCpu.docIdx << " " << pairCpu.score << endl;
            cout << "CUDA: " << pairCuda.reqIdx << " " << pairCuda.docIdx << " " << pairCuda.score << endl;
            throw runtime_error("Mismatch detected!");
        }

        if (pairCpu.reqIdx != pairCusparse.reqIdx ||
            pairCpu.docIdx != pairCusparse.docIdx ||
            abs(pairCpu.score - pairCusparse.score) > 1e-3)
        {
            cout << "Mismatch at pairIdx " << pairIdx << endl;
            cout << "CPU: " << pairCpu.reqIdx << " " << pairCpu.docIdx << " " << pairCpu.score << endl;
            cout << "CUSPARSE: " << pairCusparse.reqIdx << " " << pairCusparse.docIdx << " " << pairCusparse.score << endl;
            throw runtime_error("Mismatch detected!");
        }
    }
    cout << "All results match!" << endl;
}

int main()
{
    Setting setting;
    setting.numTrials = kNumTrials;
    
    Data data = genData();

    methodCpu(data, setting);
    methodCuda(data, setting);
    methodCusparse(data, setting);

    checkData(data);

    data.free();

    return 0;
}