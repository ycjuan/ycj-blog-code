#pragma once

#include "common/typedef.hpp"

struct DensificationTask
{
    T_DOC_IDX* d_docIdxMap;
    size_t numTasks;
    size_t embIdxBeginIncl;
    size_t embIdxEndExcl;
    T_EMB* d_workingSetEmbIndex;
};