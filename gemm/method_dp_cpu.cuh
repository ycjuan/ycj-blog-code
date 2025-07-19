#ifndef METHOD_DP_CPU_CUH
#define METHOD_DP_CPU_CUH

#include "data.cuh"

using namespace std;

void methodDpCpu(Data data)
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
            data.h_rst_dp_cpu[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstDpCpuMemLayout)] = (half)sum;
        }
    }
    cout << "DP-CPU time: " << timer.tocMs() << " ms" << endl;
}

#endif