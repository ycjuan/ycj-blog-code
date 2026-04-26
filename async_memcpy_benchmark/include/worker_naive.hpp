#pragma once

#include "worker.hpp"

class WorkerNaive : public Worker
{
public:
    WorkerNaive(int maxNumDocs, int embDim);

    void update(const std::vector<long>& jobIds, const std::vector<std::vector<T_EMB>>& embData2D) override;

    std::vector<std::vector<long>> score(const std::vector<std::vector<T_EMB>>& reqEmb, int k) const override;
};
