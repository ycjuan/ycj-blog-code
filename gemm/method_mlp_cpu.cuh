#ifndef METHOD_MLP_CPU_CUH
#define METHOD_MLP_CPU_CUH

#include "data.cuh"

using namespace std;

void methodMlpCpu(Data data)
{
    Timer timer;
    timer.tic();
    #pragma omp parallel for
    for (int docIdx = 0; docIdx < data.numDocs; docIdx++)
    {
        for (int reqIdx = 0; reqIdx < data.numReqs; reqIdx++)
        {
            // -------------
            // Step 1: Perform element-wise hadamard product. The output is a vector of embDim.
            vector<float> hprod(data.embDim, 0.0f);
            for (int embIdx = 0; embIdx < data.embDim; embIdx++)
            {
                T reqVal = data.d_req[getMemAddr(reqIdx, embIdx, data.numReqs, data.embDim, data.reqMemLayout)];
                T docVal = data.d_doc[getMemAddr(docIdx, embIdx, data.numDocs, data.embDim, data.docMemLayout)];
                hprod[embIdx] = (float)reqVal * (float)docVal;
            }

            // -------------
            // Step 2: Perform the first hidden layer transformation: hprod * wa
            // hprod is embDim x 1, wa is embDim x hiddenDim
            // The output is a vector of hiddenDim.
            vector<float> hiddenLayer(data.hiddenDim, 0.0f);
            for (int hiddenIdx = 0; hiddenIdx < data.hiddenDim; hiddenIdx++)
            {
                float sum = 0.0f;
                for (int embIdx = 0; embIdx < data.embDim; embIdx++)
                {
                    T waVal = data.d_wa[getMemAddr(embIdx, hiddenIdx, data.embDim, data.hiddenDim, data.waLayout)];
                    sum += hprod[embIdx] * (float)waVal;
                }
                sum = 1.0f / (1.0f + expf(-sum));
                hiddenLayer[hiddenIdx] = sum;
            }

            // -------------
            // Step 3: Perform the second hidden layer transformation: hiddenLayer * wb
            // hiddenLayer is hiddenDim x 1, wb is hiddenDim x 1
            // The output is a scalar.
            float rst = 0.0f;
            for (int hiddenIdx = 0; hiddenIdx < data.hiddenDim; hiddenIdx++)
            {
                T wbVal = data.d_wb[getMemAddr(hiddenIdx, 0, data.hiddenDim, 1, data.wbLayout)];
                rst += hiddenLayer[hiddenIdx] * (float)wbVal;
            }
            rst = 1.0f / (1.0f + expf(-rst));

            data.h_rst_cpu[getMemAddr(docIdx, reqIdx, data.numDocs, data.numReqs, data.rstLayoutCpu)] = (T)rst;
        }
    }
    cout << "CPU time: " << timer.tocMs() << " ms" << endl;
}

#endif