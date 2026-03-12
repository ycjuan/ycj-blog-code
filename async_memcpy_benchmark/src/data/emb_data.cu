#include "data/emb_data.hpp"
#include "utils/util.hpp"

#include <random>
#include <vector>

EmbData::EmbData(int numDocs, int embDim)
    : m_numDocs(numDocs)
    , m_embDim(embDim)
    , m_data(numDocs * embDim, "EmbData")
{
    const int totalSize = numDocs * embDim;

    std::default_random_engine rng;
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<T_EMB> hostData(totalSize);
    for (auto& val : hostData)
    {
        val = static_cast<T_EMB>(dist(rng));
    }

    CHECK_CUDA(cudaMemcpy(m_data.data(), hostData.data(), totalSize * sizeof(T_EMB), cudaMemcpyHostToDevice));

    std::uniform_int_distribution<long> idDist(0, std::numeric_limits<long>::max());
    for (int i = 0; i < numDocs; i++)
    {
        long docId;
        do
        {
            docId = idDist(rng);
        } while (m_docId2Idx.count(docId));
        m_docId2Idx[docId] = i;
    }
}

const T_EMB* EmbData::data() const { return m_data.data(); }
