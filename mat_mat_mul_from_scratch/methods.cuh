#pragma once

#include "data.cuh"

namespace MatMatMulFromScratch {

void methodCpu(Data& data);

void methodCublas(Data& data);

} // namespace BatchScalability