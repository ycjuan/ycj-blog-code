#include "working/working_emb_index.hpp"

WorkingEmbIndex::WorkingEmbIndex(size_t maxNumDocs, size_t totalEmbDim)
    : m_maxNumDocs(maxNumDocs)
    , m_totalEmbDim(totalEmbDim)
    , m_workingSetEmbIndex(maxNumDocs * totalEmbDim, "m_workingSetEmbIndex")
{
}

T_EMB* WorkingEmbIndex::data() const
{
    return m_workingSetEmbIndex.data();
}

void WorkingEmbIndex::setMemLayout(MemLayout memLayout)
{
    m_memLayout = memLayout;
}

MemLayout WorkingEmbIndex::getMemLayout() const
{
    return m_memLayout;
}


void WorkingEmbIndex::setEmbDimBeginIncl(size_t embDimBeginIncl)
{
    m_embDimBeginIncl = embDimBeginIncl;
}

size_t WorkingEmbIndex::getEmbDimBeginIncl() const
{
    return m_embDimBeginIncl;
}

void WorkingEmbIndex::setEmbDimEndExcl(size_t embDimEndExcl)
{
    m_embDimEndExcl = embDimEndExcl;
}

size_t WorkingEmbIndex::getEmbDimEndExcl() const
{
    return m_embDimEndExcl;
}
