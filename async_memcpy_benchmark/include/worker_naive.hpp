#pragma once

#include "worker.hpp"

class WorkerNaive : public Worker
{
public:
    WorkerNaive(int maxNumDocs, int embDim);

    void updateEmbData(const std::vector<long>& jobIds, const std::vector<std::vector<T_EMB>>& embData2D) override;
    void updateScalarData(const std::vector<long>& jobIds, const std::vector<float>& scalars) override;
};
