#include <iostream>

#include "common.cuh"
#include "util.cuh"

using namespace std;

void methodCpu(Data data, Setting setting)
{
    Timer timer;
    timer.tic();
    #pragma omp parallel for
    for (size_t pairIdx = 0; pairIdx < data.numPairsToScore; pairIdx++)
    {
        Pair &pair = data.d_PairsToScore[pairIdx];
        pair.score = 0;
        for (int k = 0; k < data.embDim; k++)
        {
            T reqVal = data.d_req[getMemAddr(pair.reqIdx, k, data.numReqs, data.embDim, data.reqMemLayout)];
            T docVal = data.d_doc[getMemAddr(pair.docIdx, k, data.numDocs, data.embDim, data.docMemLayout)];
            pair.score += (float)reqVal * (float)docVal;
        }
    }

    cout << "CPU time: " << timer.tocMs() << " ms" << endl;
}

__global__ void cudaKernel(Data data)
{
    int wid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (wid < data.numPairsToScore)
    {
        Pair &pair = data.d_PairsToScore[wid];
        pair.score = 0;
        for (int k = 0; k < data.embDim; k++)
        {
            T reqVal = data.d_req[getMemAddr(pair.reqIdx, k, data.numReqs, data.embDim, data.reqMemLayout)];
            T docVal = data.d_doc[getMemAddr(pair.docIdx, k, data.numDocs, data.embDim, data.docMemLayout)];
            pair.score += float(reqVal * docVal);
        }
    }
}

void methodCuda(Data data, Setting setting)
{
    int blockSize = 512;
    int gridSize = data.numPairsToScore / blockSize;
    CudaTimer timer;
    for (int t = -3; t < setting.numTrials; t++)
    {
        if (t == 0)
            timer.tic();
        cudaKernel<<<gridSize, blockSize>>>(data);
        cudaDeviceSynchronize();
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess)
        {
            ostringstream oss;
            oss << "Kernel launch failed with error: " << cudaGetErrorString(status) << "\n";
            throw runtime_error(oss.str());
        }
    }
    cout << "Kernel time: " << timer.tocMs() / setting.numTrials << " ms" << endl;
}
