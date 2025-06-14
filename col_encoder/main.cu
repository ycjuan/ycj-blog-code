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
    auto reqDataCpu = genRandEmbData(kNumReqs, kNumFields, kEmbDimPerField);
    auto docDataCpu = genRandEmbData(kNumDocs, kNumFields, kEmbDimPerField);
    auto taskDataCpu = genRandScoringTasks(kNumReqs, kNumToScore, kNumDocs);

    // -------------------
    // Convert to GPU data
    auto reqDataGpu = convertEmbDataToGpu(reqDataCpu);
    auto docDataGpu = convertEmbDataToGpu(docDataCpu);
    auto taskDataGpu = convertScoringTasksToGpu(taskDataCpu);
    
    // -------------------
    // Run scoring
    colEncoderScorerCpu(reqDataCpu, docDataCpu, taskDataCpu);
    colEncoderScorerGpu(reqDataGpu, docDataGpu, taskDataGpu);

    // -------------------
    // Compare results
    compareResults(taskDataCpu, taskDataGpu);

    // -------------------
    // Test latency
    Timer timer;
    for (int trial = -3; trial < kNumTrials; ++trial) 
    {
        if (trial == 0) 
        {
            timer.tic();
        }
        colEncoderScorerGpu(reqDataGpu, docDataGpu, taskDataGpu);
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
    const int kNumDocs = 50000;
    const int kNumFields = 10;
    const int kEmbDimPerField = 512;
    const int kNumToScore = 1000;

    runTest(kNumReqs, kNumDocs, kNumFields, kEmbDimPerField, kNumToScore, 20);

    return 0;
}