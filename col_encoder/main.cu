#include <sstream>
#include <iostream>

#include "col_encoder_gpu.cuh"
#include "col_encoder_cpu.cuh"
#include "data_struct.cuh"
#include "data_cvt.cuh"
#include "data_gen.cuh"
#include "util.cuh"

using namespace std;

void compareResults(const std::vector<ScoringTask>& cpuTasks, const ScoringTasksGpu& gpuTasksRaw)
{
    using namespace std;

    // Copy results back from GPU
    vector<ScoringTask> gpuTasks = convertScoringTasksBackToCpu(gpuTasksRaw);

    // Compare CPU and GPU results
    for (size_t i = 0; i < cpuTasks.size(); ++i) 
    {
        float cpuResult = cpuTasks[i].result;
        float gpuResult = gpuTasks[i].result;
        float relativeError = abs(cpuResult - gpuResult) / (abs(cpuResult) + 1e-6f); // Avoid division by zero
        if (relativeError > 1e-3) // Use relative error for comparison
        {
            ostringstream oss;
            oss << "Mismatch at task " << i << ": CPU result = " << cpuTasks[i].result 
                << ", GPU result = " << gpuTasks[i].result;
            throw runtime_error(oss.str());
        }
    }
}

struct TestParams
{
    int kNumReqs;
    int kNumDocs;
    int kNumFields;
    int kEmbDimPerField;
    int kNumToScore;
    float kH2DRatio;
    int kNumTrials;
};

void runTest(const TestParams& params,
             const vector<vector<vector<float>>> &docDataCpu,
             const EmbData &docDataGpu)
{
    using namespace std;

    // -------------------
    // Print test parameters
    cout << "kNumReqs: " << params.kNumReqs << ", "
         << "kNumDocs: " << params.kNumDocs << ", "
         << "kNumFields: " << params.kNumFields << ", "
         << "kEmbDimPerField: " << params.kEmbDimPerField << ", "
         << "kNumToScore: " << params.kNumToScore << ", "
         << "kH2DRatio: " << params.kH2DRatio << endl;

    // -------------------
    // Random data CPU
    auto reqDataCpu = genRandEmbData(params.kNumReqs, params.kNumFields, params.kEmbDimPerField);
    auto taskDataCpu = genRandScoringTasks(params.kNumReqs, params.kNumToScore, params.kNumDocs);

    // -------------------
    // Convert to GPU data
    auto reqDataGpu = convertEmbDataToGpu(reqDataCpu);
    auto taskDataGpu = convertScoringTasksToGpu(taskDataCpu);
    
    // -------------------
    // Run scoring
    colEncoderScorerCpu(reqDataCpu, docDataCpu, taskDataCpu);
    colEncoderScorerGpu(reqDataGpu, docDataGpu, taskDataGpu, params.kH2DRatio);

    // -------------------
    // Compare results
    compareResults(taskDataCpu, taskDataGpu);

    // -------------------
    // Test latency
    double timeMsTotalSum = 0.0;
    double timeMsCopySum = 0.0;
    double timeMsScoringSum = 0.0;
    for (int trial = -3; trial < params.kNumTrials; ++trial) 
    {
        auto rst = colEncoderScorerGpu(reqDataGpu, docDataGpu, taskDataGpu, params.kH2DRatio);
        if (trial >= 0) 
        {
            timeMsTotalSum += rst.totalTimeMs;
            timeMsCopySum += rst.copyTimeMs;
            timeMsScoringSum += rst.scoringTimeMs;
        }
    }
    float latencyMsTotal = (timeMsTotalSum / params.kNumTrials);
    float latencyMsCopy = (timeMsCopySum / params.kNumTrials);
    float latencyMsScoring = (timeMsScoringSum / params.kNumTrials);
    cout << "Average latency per trial: " << latencyMsTotal << " ms" << endl;
    cout << "Average copy latency per trial: " << latencyMsCopy << " ms" << endl;
    cout << "Average scoring latency per trial: " << latencyMsScoring << " ms" << endl;

    // -------------------
    // Compare results just in case
    compareResults(taskDataCpu, taskDataGpu);

    // -------------------
    // Free GPU data
    reqDataGpu.free();
    taskDataGpu.free();
}

int main() 
{
    const int kNumTrials = 50;
    const int kNumDocs = 50000;
    const int kNumFields = 10;
    const int kEmbDimPerField = 512;
    const std::vector<int> kNumReqLists = {4};
    const vector<int> kNumToScoreList = {4000};
    const vector<float> kH2DRatioList = {0, 0.25, 0.5, 0.75, 1.0};

    // -------------------
    // DocSize data
    auto docDataCpu = genRandEmbData(kNumDocs, kNumFields, kEmbDimPerField);
    auto docDataGpu = convertEmbDataToGpu(docDataCpu);

    // -------------------
    // Run tests for different numbers of requests and documents to score
    for (const int kNumReqs : kNumReqLists) 
    {
        for (const int kNumToScore : kNumToScoreList) 
        {
            for (const float kH2DRatio : kH2DRatioList)
            {
                TestParams params;
                params.kNumReqs = kNumReqs;
                params.kNumDocs = kNumDocs;
                params.kNumFields = kNumFields;
                params.kEmbDimPerField = kEmbDimPerField;
                params.kNumToScore = kNumToScore;
                params.kH2DRatio = kH2DRatio;
                params.kNumTrials = kNumTrials;
                runTest(params, docDataCpu, docDataGpu);
            }
        }
    }

    // -------------------
    // Free GPU data
    docDataGpu.free();

    return 0;
}