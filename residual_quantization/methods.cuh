#pragma once

#include "data.cuh"

enum class Method
{
    REFERENCE = 0,
    BASELINE_H2D = 1,
    BASELINE_D2D = 2,
    RES_QUANT_H2D = 3,
    RES_QUANT_D2D = 4
};

void runMethod(Data data, Method method);