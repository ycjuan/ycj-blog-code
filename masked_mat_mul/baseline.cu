#include <iostream>

#include "common.cuh"
#include "util.cuh"

using namespace std;

void methodCpu(Data data, Setting setting)
{
    Timer timer;
    timer.tic();
    #pragma omp parallel for
    for (int i = 0; i < data.numDocs; i++)
    {
        for (int j = 0; j < data.numReqs; j++)
        {
            float sum = 0;
            for (int k = 0; k < data.embDim; k++)
            {
                T reqVal = data.d_req[getMemAddr(j, k, data.numReqs, data.embDim, data.reqMemLayout)];
                T docVal = data.d_doc[getMemAddr(i, k, data.numDocs, data.embDim, data.docMemLayout)];
                sum += (float)reqVal * (float)docVal;
            }
            data.h_rst_cpu[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutCpu)] = (half)sum;
        }
    }
    cout << "CPU time: " << timer.tocMs() << " ms" << endl;
}

__global__ void cudaKernel(Data data)
{
    int threadId = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int i = threadId / data.numReqs;
    int j = threadId % data.numReqs;

    if (i < data.numDocs && j < data.numReqs)
    {
        float sum = 0;
        for (int k = 0; k < data.embDim; k++)
        {
            T reqVal = data.d_req[getMemAddr(j, k, data.numReqs, data.embDim, data.reqMemLayout)];
            T docVal = data.d_doc[getMemAddr(i, k, data.numDocs, data.embDim, data.docMemLayout)];            
            sum += float(reqVal * docVal);
        }
        data.d_rst_kernel[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutGpuKernel)] = sum;
    }
}

void methodCuda(Data data, Setting setting)
{
    int blockSize = 512;
    int gridSize = size_t(data.numDocs) * data.numReqs / blockSize;
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
