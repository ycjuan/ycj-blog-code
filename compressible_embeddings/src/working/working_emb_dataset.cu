#include "working/working_emb_dataset.hpp"

WorkingEmbDataset::WorkingEmbDataset(int maxNumDocs, int totalEmbDim)
    : m_maxNumDocs(maxNumDocs)
    , m_totalEmbDim(totalEmbDim)
    , m_workingEmbDataset(maxNumDocs * totalEmbDim, "m_workingSetEmbDataset")
{
}

T_EMB* WorkingEmbDataset::data() const
{
    return m_workingEmbDataset.data();
}

void WorkingEmbDataset::setMemLayout(MemLayout memLayout)
{
    m_memLayout = memLayout;
}

MemLayout WorkingEmbDataset::getMemLayout() const
{
    return m_memLayout;
}


void WorkingEmbDataset::setEmbDimBeginIncl(int embDimBeginIncl)
{
    m_embDimBeginIncl = embDimBeginIncl;
}

int WorkingEmbDataset::getEmbDimBeginIncl() const
{
    return m_embDimBeginIncl;
}

void WorkingEmbDataset::setEmbDimEndExcl(int embDimEndExcl)
{
    m_embDimEndExcl = embDimEndExcl;
}

int WorkingEmbDataset::getEmbDimEndExcl() const
{
    return m_embDimEndExcl;
}
