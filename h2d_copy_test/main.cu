#include <iostream>
#include <stdexcept>
#include <vector>

#include "util.cuh"
#include "tasks.cuh"

int main()
{
    //printDeviceInfo();

    CudaCoreMatMatMulRunner runner(1024, 1024, 1024);
    runner.run();

    TensorCoreMatMatMulRunner tensorCoreRunner(1024, 1024, 1024);
    tensorCoreRunner.run();

    return 0;
}