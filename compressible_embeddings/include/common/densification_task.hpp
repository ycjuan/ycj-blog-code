#pragma once

#include "common/typedef.hpp"

struct CopyTask
{
    T_DOC_IDX srcDocIdx;
    T_DOC_IDX dstDocIdx;
};

struct DensificationTask
{
    size_t numDocsToDensify;
    size_t numCopyTasks;
    size_t globalEmbIdxBeginIncl;
    size_t globalEmbIdxEndExcl;
    T_EMB* d_workingEmbDataset;
    CopyTask* d_copyTasks;
};