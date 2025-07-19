#ifndef METHOD_DP_CPU_CUH
#define METHOD_DP_CPU_CUH

#include "data.cuh"

using namespace std;

void methodDpCpu(Data data)
{
    Timer timer;
    timer.tic();
    #pragma omp parallel for
    for (int docIdx = 0; docIdx < data.numDocs; docIdx++)
    {
        for (int reqIdx = 0; reqIdx < data.numReqs; reqIdx++)
        {
            float sum = 0;
            for (int embIdx = 0; embIdx < data.embDim; embIdx++)
            {
                T reqVal = data.d_req[getMemAddr(reqIdx, embIdx, data.numReqs, data.embDim, data.reqMemLayout)];
                T docVal = data.d_doc[getMemAddr(docIdx, embIdx, data.numDocs, data.embDim, data.docMemLayout)];
                sum += (float)reqVal * (float)docVal;
            }
            data.h_rst_dp_cpu[getMemAddr(docIdx, reqIdx, data.numDocs, data.numReqs, data.rstDpCpuMemLayout)] = (half)sum;
        }
    }
    cout << "DP-CPU time: " << timer.tocMs() << " ms" << endl;
}

#endif