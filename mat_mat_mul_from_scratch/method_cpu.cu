#include "methods.cuh"

namespace MatMatMulFromScratch
{

void methodCpu(Data& data)
{
    #pragma omp parallel for
    for (int m = 0; m < data.M; m++)
    {
        for (int n = 0; n < data.N; n++)
        {
            float rst = 0;
            for (int k = 0; k < data.K; k++)
            {
                T a = data.d_A[getMemAddrA(m, k, data.M, data.K)];
                T b = data.d_B[getMemAddrB(k, n, data.K, data.N)];
                rst += static_cast<float>(a) * static_cast<float>(b);
            }
            data.h_C[getMemAddrC(m, n, data.M, data.N)] = rst;
        }
    }
}

} // namespace BatchScalability