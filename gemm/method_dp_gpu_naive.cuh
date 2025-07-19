#ifndef METHOD_DP_GPU_NAIVE_CUH
#define METHOD_DP_GPU_NAIVE_CUH

#include <iostream>

#include "data.cuh"

using namespace std;

__global__ void matMul(Data data)
{
    int taskIdx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int i = taskIdx / data.numReqs;
    int j = taskIdx % data.numReqs;

    if (i < data.numDocs && j < data.numReqs)
    {
        float sum = 0;
        for (int k = 0; k < data.embDim; k++)
        {
            T reqVal = data.d_req[getMemAddr(j, k, data.numReqs, data.embDim, data.reqMemLayout)];
            T docVal = data.d_doc[getMemAddr(i, k, data.numDocs, data.embDim, data.docMemLayout)];            
            sum += float(reqVal * docVal);
        }
        data.d_rst_dp_gpu_naive[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstDpGpuNaiveMemLayout)] = sum;
    }
}

void methodDpGpuNaive(Data data, int numTrials)
{
    int blockSize = 512;
    int gridSize = (data.numDocs * data.numReqs + blockSize - 1) / blockSize;
    CudaTimer timer;
    for (int t = -3; t < numTrials; t++)
    {
        if (t == 0)
            timer.tic();
        matMul<<<gridSize, blockSize>>>(data);
        cudaDeviceSynchronize();
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess)
        {
            ostringstream oss;
            oss << "Kernel launch failed with error: " << cudaGetErrorString(status) << "\n";
            throw runtime_error(oss.str());
        }
    }
    cout << "DP-Naive time: " << timer.tocMs() / numTrials << " ms" << endl;
}

#endif