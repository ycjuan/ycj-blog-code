#include <sstream>
#include <iostream>

#include "col_enc_gpu.cuh"
#include "col_enc_cpu.cuh"
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
    auto reqDataCpu = genRandColEncData(kNumReqs, kNumFields, kEmbDimPerField);
    auto docDataCpu = genRandColEncData(kNumDocs, kNumFields, kEmbDimPerField);
    auto taskDataCpu = genRandScoringTasks(kNumReqs, kNumToScore, kNumDocs);

    // -------------------
    // Convert to GPU data
    auto reqDataGpu = convertColEncDataToGpu(reqDataCpu);
    auto docDataGpu = convertColEncDataToGpu(docDataCpu);
    auto taskDataGpu = convertScoringTasksToGpu(taskDataCpu);

    // Malloc buffer
    EMB_T* d_tmpDocData = nullptr;
    size_t size_byte_tmpDocData = (size_t)kNumToScore * docDataGpu.numFields * docDataGpu.embDimPerField * sizeof(EMB_T);
    cudaError_t cudaError = cudaMalloc(&d_tmpDocData, size_byte_tmpDocData);
    if (cudaError != cudaSuccess)
    {
        throw runtime_error("Failed to allocate device memory for d_tmpDocData: " + std::to_string(cudaError));
    }

    float *d_tmpRst = nullptr;
    size_t size_byte_tmpRst = (size_t)kNumToScore * docDataGpu.numFields * sizeof(float);
    cudaError = cudaMalloc(&d_tmpRst, size_byte_tmpRst);
    if (cudaError != cudaSuccess)
    {
        throw runtime_error("Failed to allocate device memory for d_tmpRst: " + std::to_string(cudaError));
    }

    // -------------------
    // Run scoring
    colEncScorerCpu(reqDataCpu, docDataCpu, taskDataCpu);
    // Create cuBLAS handle
    cublasHandle_t cublasHandle;
    cublasStatus_t cublasStatus = cublasCreate(&cublasHandle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) 
    {
        throw runtime_error("Failed to create cuBLAS handle");
    }
    colEncScorerGpu(reqDataGpu, docDataGpu, taskDataGpu, 0, cublasHandle, d_tmpDocData, d_tmpRst);

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
        colEncScorerGpu(reqDataGpu, docDataGpu, taskDataGpu, 0, cublasHandle, d_tmpDocData, d_tmpRst);
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
    cudaFree(d_tmpDocData);
    cudaFree(d_tmpRst);
    cublasStatus = cublasDestroy(cublasHandle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) 
    {
        throw runtime_error("Failed to destroy cuBLAS handle: " + std::to_string(cublasStatus));
    }
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