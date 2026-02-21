#pragma once

#include "common/typedef.hpp"

struct DensificationTask
{
    T_DOC_IDX* d_docIdxList;
    size_t numDocsToDensify;
    size_t globalEmbIdxBeginIncl;
    size_t globalEmbIdxEndExcl;
    T_EMB* d_workingEmbIndex;
    int8_t* hp_isCached;
};