#include "util.cuh"
#include <bits/types/struct_sched_param.h>
#include <iostream>


int main()
{
    ParallelizationWithCudaStream::printDeviceInfo();
        
    std::cout << "Start" << std::endl;
    ParallelizationWithCudaStream::Timer timer;
    timer.tic();
    ParallelizationWithCudaStream::printDeviceInfo();

    std::cout << "Hello, World!" << std::endl;

    return 0;
}