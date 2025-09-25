#pragma once

#include "data.cuh"

namespace BatchScalability {

void methodCpu(Data& data);

void methodGpuNaive1(Data& data);

void methodGpuNaive2(Data& data);

void methodGpuNaive3(Data& data);

} // namespace BatchScalability