#include "data/emb_data.hpp"

#include <random>
#include <vector>

EmbData::EmbData(int numDocs, int embDim)
    : numDocs(numDocs)
    , embDim(embDim)
    , data(numDocs * embDim, "EmbData")
{
    const int totalSize = numDocs * embDim;

    std::default_random_engine rng;
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<T_EMB> hostData(totalSize);
    for (auto& val : hostData) {
        val = __float2bfloat16(dist(rng));
    }

    cudaMemcpy(data.data(), hostData.data(), totalSize * sizeof(T_EMB), cudaMemcpyHostToDevice);
}
