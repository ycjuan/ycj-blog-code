#pragma once

#include "common/typedef.hpp"
#include "utils/cuda_raii.hpp"

#include <unordered_map>

class EmbData
{
public:
    EmbData(int numDocs, int embDim);

    const T_EMB* data() const;

private:
    int numDocs;
    int embDim;
    CudaDeviceArray<T_EMB> d_data;
    std::unordered_map<long, int> docId2Idx;
};
