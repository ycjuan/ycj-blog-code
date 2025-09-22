#include "methods.cuh"

namespace BatchScalability
{

void methodCpu(Data& data)
{
    for (int reqIdx = 0; reqIdx < data.numReqs; reqIdx++)
    {
        for (int docIdx = 0; docIdx < data.numDocs; docIdx++)
        {
            float rst = 0;
            for (int embIdx = 0; embIdx < data.embDim; embIdx++)
            {
                float reqVal = data.d_reqData[getMemAddrReq(reqIdx, embIdx, data.numReqs, data.embDim)];
                float docVal = data.d_docData[getMemAddrDoc(docIdx, embIdx, data.numDocs, data.embDim)];
                rst += std::sqrt(reqVal * docVal);
            }
            data.h_rstDataCpu[getMemAddrRst(reqIdx, docIdx, data.numReqs, data.numDocs)] = rst;
        }
    }
}

} // namespace BatchScalability