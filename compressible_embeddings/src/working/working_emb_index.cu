#include "working/working_emb_index.hpp"

WorkingEmbDataset::WorkingEmbDataset(size_t maxNumDocs, size_t totalEmbDim)
    : m_maxNumDocs(maxNumDocs)
    , m_totalEmbDim(totalEmbDim)
    , m_workingEmbIndex(maxNumDocs * totalEmbDim, "m_workingSetEmbIndex")
{
}

T_EMB* WorkingEmbDataset::data() const
{
    return m_workingEmbIndex.data();
}

void WorkingEmbDataset::setMemLayout(MemLayout memLayout)
{
    m_memLayout = memLayout;
}

MemLayout WorkingEmbDataset::getMemLayout() const
{
    return m_memLayout;
}


void WorkingEmbDataset::setEmbDimBeginIncl(size_t embDimBeginIncl)
{
    m_embDimBeginIncl = embDimBeginIncl;
}

size_t WorkingEmbDataset::getEmbDimBeginIncl() const
{
    return m_embDimBeginIncl;
}

void WorkingEmbDataset::setEmbDimEndExcl(size_t embDimEndExcl)
{
    m_embDimEndExcl = embDimEndExcl;
}

size_t WorkingEmbDataset::getEmbDimEndExcl() const
{
    return m_embDimEndExcl;
}
