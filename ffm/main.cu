#include <sstream>
#include <iostream>

#include "ffm_gpu.cuh"
#include "ffm_cpu.cuh"
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
        if (abs(cpuTasks[i].result - gpuTasks[i].result) > 1e-3) 
        {
            ostringstream oss;
            oss << "Mismatch at task " << i << ": CPU result = " << cpuTasks[i].result 
                << ", GPU result = " << gpuTasks[i].result;
            throw runtime_error(oss.str());
        }
    }
}

void runTest(const int kNumReqs, const int kNumDocs, const int kNumFields, const int kEmbDimPerField, const int kNumToScore, const int kNumTrials)
{
    using namespace std;

    // -------------------
    // Print test parameters
    cout << "kNumReqs: " << kNumReqs << ", "
         << "kNumDocs: " << kNumDocs << ", "
         << "kNumFields: " << kNumFields << ", "
         << "kEmbDimPerField: " << kEmbDimPerField << ", "
         << "kNumToScore: " << kNumToScore << endl;

    // -------------------
    // Random data CPU
    auto reqDataCpu = genRandFFMData(kNumReqs, kNumFields, kEmbDimPerField);
    auto docDataCpu = genRandFFMData(kNumDocs, kNumFields, kEmbDimPerField);
    auto taskDataCpu = genRandScoringTasks(kNumReqs, kNumToScore, kNumDocs);

    // -------------------
    // Convert to GPU data
    auto reqDataGpu = convertFFMDataToGpu(reqDataCpu);
    auto docDataGpu = convertFFMDataToGpu(docDataCpu);
    auto taskDataGpu = convertScoringTasksToGpu(taskDataCpu);
    
    // -------------------
    // Run scoring
    ffmScorerCpu(reqDataCpu, docDataCpu, taskDataCpu);
    ffmScorerGpu(reqDataGpu, docDataGpu, taskDataGpu);

    // -------------------
    // Compare results
    compareResults(taskDataCpu, taskDataGpu);

    // -------------------
    // Test latency
    Timer timer;
    timer.tic();
    for (int trial = 0; trial < kNumTrials; ++trial) 
    {
        ffmScorerGpu(reqDataGpu, docDataGpu, taskDataGpu);
    }
    float latencyMs = timer.tocMs() / kNumTrials;
    cout << "Average latency per trial: " << latencyMs << " ms" << endl;

    // -------------------
    // Compare results just in case
    compareResults(taskDataCpu, taskDataGpu);

    // -------------------
    // Free GPU data
    reqDataGpu.free();
    docDataGpu.free();
    taskDataGpu.free();
}

int main() 
{
    const int kNumReqs = 16;
    const int kNumDocs = 1000;
    const int kNumFields = 10;
    const int kEmbDimPerField = 8;
    const int kNumToScore = 5;

    runTest(kNumReqs, kNumDocs, kNumFields, kEmbDimPerField, kNumToScore, 100);

    return 0;
}