#pragma once

#include "worker.hpp"

class WorkerNaive : public Worker
{
public:
    WorkerNaive(int maxNumDocs, int embDim);

    void update(const std::vector<long>& jobIds, const std::vector<std::vector<T_EMB>>& embData2D) override;
};
