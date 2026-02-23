#pragma once

#include <vector>

#include "common/memory_layout.hpp"
#include "common/typedef.hpp"

struct CopyTask
{
    T_DOC_IDX srcDocIdx;
    T_DOC_IDX dstDocIdx;
};

struct DensificationTask
{
    // Caller-provided
    std::vector<T_DOC_IDX> desiredDocIdxList;
    int globalEmbIdxBeginIncl;
    int globalEmbIdxEndExcl;
    MemLayout memLayout;

    // Filled in by manager
    int numCopyTasks;
    T_EMB* d_workingEmbDataset;
    CopyTask* d_copyTasks;
};