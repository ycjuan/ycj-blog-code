#pragma once

#include "common/typedef.hpp"
#include "utils/cuda_raii.hpp"

class EmbData {
public:
    EmbData(int numDocs, int embDim);

private:
    int numDocs;
    int embDim;
    CudaDeviceArray<T_EMB> data;
};
