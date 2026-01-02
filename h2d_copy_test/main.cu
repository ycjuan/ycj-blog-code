#include <iostream>
#include <stdexcept>
#include <vector>

#include "util.cuh"
#include "tasks.cuh"

int main()
{
    constexpr int m = 1024 * 1024;
    constexpr int n = 64;
    constexpr int k = 256;

    CudaCoreMatMatMulRunner runner(m, n, k);
    runner.run();

    TensorCoreMatMatMulRunner tensorCoreRunner(m, n, k);
    tensorCoreRunner.run();

    H2DMemcpyRunner h2dMemcpyRunner(m, n, k);
    h2dMemcpyRunner.run();

    return 0;
}