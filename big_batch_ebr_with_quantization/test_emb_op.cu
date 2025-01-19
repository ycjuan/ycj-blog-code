#include <iostream>
#include <vector>

#include "emb_data_struct.cuh"
#include "emb_op.cuh"

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

void checkRst(EmbData data)
{
    vector<float> rstGpu((size_t)data.numDocs * data.numReqs);
    CHECK_CUDA(cudaMemcpy(rstGpu.data(), data.d_rst, (size_t)data.numDocs * data.numReqs * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < data.numDocs; i++)
    {
        for (int j = 0; j < data.numReqs; j++)
        {
            float cpuVal = data.h_rst[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstMemLayout)];
            float gpuVal = rstGpu[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstMemLayout)];

            if (abs(cpuVal - gpuVal) / abs(cpuVal) > 1e-3)
            {
                cout << "GPU error at (" << i << ", " << j << "): " << cpuVal << " != " << gpuVal << endl;
                return;
            }
        }
    }
    cout << "Test passed!!!" << endl;
}

int main()
{
    int kNumDocs = 1000000;
    int kNumReqs = 16;
    int kEmbDim = 128;
    int kTrials = 100;
    bool kSkipVerification = false;

    EmbData data;
    cout << "Initializing data..." << endl;
    data.initRand(kNumDocs, kNumReqs, kEmbDim, ROW_MAJOR, ROW_MAJOR);
    // Note. It looks like cublasGemmEx did some shape checking, so every combination of mem layouts have similar performance.

    cout << "Running embOpGpu..." << endl;
    double timeMsCuBlasSum = 0;
    for (int i = -3; i < kTrials; i++)
    {
        embOpGpu(data);
        if (i >= 0)        
            timeMsCuBlasSum += data.timeMsCuBlas;
    }
    cout << "Average time for embOpGpu: " << timeMsCuBlasSum / kTrials << " ms" << endl;

    if (!kSkipVerification)
    {
        cout << "Running embOpCpu..." << endl;
        embOpCpu(data);

        cout << "Checking results..." << endl;
        checkRst(data);
    }
 
    return 0;
}