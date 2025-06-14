#ifndef COL_ENCODER_CPU_CUH
#define COL_ENCODER_CPU_CUH

#include <vector>
#include <stdexcept>
#include <limits>

#include "data_struct.cuh"

void colEncoderScorerCpu(const std::vector<std::vector<std::vector<float>>> &reqData,
                         const std::vector<std::vector<std::vector<float>>> &docData,
                         std::vector<ScoringTask> &tasks)
{
    int numFields = reqData.at(0).size();
    int embDimPerField = reqData.at(0).at(0).size();

    // Process each task
    for (int taskIdx = 0; taskIdx < tasks.size(); ++taskIdx) 
    {
        ScoringTask &task = tasks[taskIdx];

        task.result = 0.0f;

        for (int reqFieldIdx = 0; reqFieldIdx < numFields; ++reqFieldIdx) 
        {
            float maxSim = std::numeric_limits<float>::lowest();
            for (int docFieldIdx = 0; docFieldIdx < numFields; ++docFieldIdx) 
            {
                float sim = 0.0f;
                for (int embIdx = 0; embIdx < embDimPerField; ++embIdx) 
                {
                    float reqVal = reqData.at(task.reqIdx).at(reqFieldIdx).at(embIdx);
                    float docVal = docData.at(task.docIdx).at(docFieldIdx).at(embIdx);
                    sim += reqVal * docVal;
                }
                maxSim = std::max(maxSim, sim);
            }
            task.result += maxSim;
        }
    }
}

#endif // COL_ENCODER_CPU_CUH