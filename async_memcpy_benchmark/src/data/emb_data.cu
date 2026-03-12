#include "data/emb_data.hpp"
#include "utils/util.hpp"

#include <vector>

EmbData::EmbData(int maxNumDocs, int embDim)
    : m_maxNumDocs(maxNumDocs)
    , m_embDim(embDim)
    , m_data(maxNumDocs * embDim, "EmbData")
{
}

const T_EMB* EmbData::data() const { return m_data.data(); }
