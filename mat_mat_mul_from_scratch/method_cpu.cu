#include "methods.cuh"

namespace MatMatMulFromScratch
{

void methodCpu(Data& data)
{
    //#pragma omp parallel for
    for (int reqIdx = 0; reqIdx < data.M; reqIdx++)
    {
        for (int docIdx = 0; docIdx < data.N; docIdx++)
        {
            double rst = 0;
            for (int embIdx = 0; embIdx < data.K; embIdx++)
            {
                float reqVal = data.d_B[getMemAddrA(reqIdx, embIdx, data.M, data.K)];
                float docVal = data.d_A[getMemAddrB(docIdx, embIdx, data.N, data.K)];
                rst += std::sqrt(reqVal * docVal);
            }
            data.h_C[getMemAddrC(reqIdx, docIdx, data.M, data.N)] = rst;
        }
    }
}

} // namespace BatchScalability