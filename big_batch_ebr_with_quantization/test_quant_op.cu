#include <iostream>
#include <vector>
#include <cassert>

#include "quant_data_struct.cuh"
#include "quant_op.cuh"

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

void checkRst(QuantData data)
{
    vector<T_QUANT_RST> rstGpu((size_t)data.numDocs * data.numReqs);
    CHECK_CUDA(cudaMemcpy(rstGpu.data(), data.d_rstGpu, (size_t)data.numDocs * data.numReqs * sizeof(T_QUANT_RST), cudaMemcpyDeviceToHost));
    for (int i = 0; i < data.numDocs; i++)
    {
        for (int j = 0; j < data.numReqs; j++)
        {
            float cpuVal = data.h_rstCpu[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutCpu)];
            float gpuVal = rstGpu[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstLayoutGpu)];

            if (cpuVal != gpuVal)
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
    int kNumDocs = 3200000;
    int kNumReqs = 32;
    int kNumInt32 = 4;
    int kTrials = 100;
    bool kSkipVerification = false;

    assert(kNumDocs % 32 == 0);
    assert(kNumReqs % 32 == 0);
    assert(kNumInt32 % 4 == 0);

    QuantData data;
    cout << "Initializing data..." << endl;
    data.initRand(kNumDocs, kNumReqs, kNumInt32);

    cout << "Running quantOpGPU..." << endl;
    double timeMsGpu = 0;
    for (int i = -3; i < kTrials; i++)
    {
        quantOpGpu(data);
        if (i >= 0)        
            timeMsGpu += data.timeMsGpu;
    }
    cout << "Average time for quantOpGPU: " << timeMsGpu / kTrials << " ms" << endl;

    if (!kSkipVerification)
    {
        cout << "Running quantOpCpu..." << endl;
        quantOpCpu(data);

        cout << "Checking results..." << endl;
        checkRst(data);
    }
 
    return 0;
}