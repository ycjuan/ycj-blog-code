#ifndef BATCH_SCALABILITY_METHODS_CUH
#define BATCH_SCALABILITY_METHODS_CUH

#include "data.cuh"

namespace BatchScalability {

void methodCpu(Data& data);

void methodGpuNaive(Data& data);

} // namespace BatchScalability

#endif