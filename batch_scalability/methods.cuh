#ifndef BATCH_SCALABILITY_METHODS_CUH
#define BATCH_SCALABILITY_METHODS_CUH

#include "data.cuh"

namespace BatchScalability {

void methodCpu(Data& data);

void methodGpuNaive1(Data& data);

void methodGpuNaive2(Data& data);

} // namespace BatchScalability

#endif