#include <sstream>

#include "ffm_gpu.cuh"
#include "data_struct.cuh"
#include "data_cvt.cuh"
#include "data_gen.cuh"

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

void runTest(const int kNumReqs, const int kNumDocs, const int kNumFields, const int kEmbDimPerField, const int kNumToScore)
{
    using namespace std;

    // -------------------
    // Random data CPU
    auto randReqDataCpu = genRandFFMData(kNumReqs, kNumFields, kEmbDimPerField);
    auto randDocDataCpu = genRandFFMData(kNumDocs, kNumFields, kEmbDimPerField);

    // -------------------
    // Random task CPU
    auto randTaskCpu = genRandScoringTasks(kNumReqs, kNumToScore, kNumDocs);

    // -------------------
    // Convert to GPU data
    auto reqDataGpu = convertFFMDataToGpu(randReqDataCpu);
    auto docDataGpu = convertFFMDataToGpu(randDocDataCpu);
    auto taskDataGpu = convertScoringTasksToGpu(randTaskCpu);
    
    // -------------------
    // Run the scoring kernel
    ffmScorer(reqDataGpu, docDataGpu, taskDataGpu);
}

int main() 
{
    const int kNumReqs = 16;
    const int kNumDocs = 1000;
    const int kNumFields = 10;
    const int kEmbDimPerField = 8;
    const int kNumToScore = 5;

    runTest(kNumReqs, kNumDocs, kNumFields, kEmbDimPerField, kNumToScore);

    return 0;
}