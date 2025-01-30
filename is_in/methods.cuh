#ifndef METHOD_CUH
#define METHOD_CUH

#include "common.cuh"

void methodCpu(Data &data, Setting setting);

void methodGpuNaive(Data &data, Setting setting);

void methodGpuBinarySearch(Data &data, Setting setting);

void methodGpuBitTrick(Data &data, Setting setting);

void methodGpuBinarySearchPlus(Data &data, Setting setting);

#endif