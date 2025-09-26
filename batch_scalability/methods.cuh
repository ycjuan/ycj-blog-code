#pragma once

#include "data.cuh"

namespace BatchScalability {

void methodCpu(Data& data);

void methodGpu1(Data& data);

void methodGpu2(Data& data);

void methodGpu3(Data& data);

void methodGpu4(Data& data);

void methodGpu5(Data& data);

} // namespace BatchScalability