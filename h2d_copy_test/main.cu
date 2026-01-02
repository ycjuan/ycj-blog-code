#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <future>
#include <algorithm>

#include "util.cuh"
#include "tasks.cuh"

std::vector<float> runTask(BaseRunner& runner, int sleepMs, int expDurationSec)
{
    std::vector<float> timeRecords;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    while (std::chrono::high_resolution_clock::now() - start < std::chrono::seconds(expDurationSec))
    {
        Timer timer;
        timer.tic();
        runner.run();
        timeRecords.push_back(timer.tocMs());
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
    }
    std::cout << runner.getName() << " completed" << std::endl;
    return timeRecords;
}

void printTimeRecords(std::vector<float> timeRecords)
{
    std::sort(timeRecords.begin(), timeRecords.end());

    std::cout << "p50: " << timeRecords[static_cast<int>(timeRecords.size() * 0.50)] << " ms" << std::endl;
    std::cout << "p90: " << timeRecords[static_cast<int>(timeRecords.size() * 0.90)] << " ms" << std::endl;
    std::cout << "p95: " << timeRecords[static_cast<int>(timeRecords.size() * 0.95)] << " ms" << std::endl;
    std::cout << "p99: " << timeRecords[static_cast<int>(timeRecords.size() * 0.99)] << " ms" << std::endl;
    std::cout << "p100: " << timeRecords.back() << " ms" << std::endl;
}

int main()
{
    constexpr int m = 1024 * 1024;
    constexpr int n = 64;
    constexpr int k = 256;
    constexpr int expDurationSec = 10;
    constexpr int cudaCoreMatMatMulSleepMs = 100;
    constexpr int tensorCoreMatMatMulSleepMs = 100;
    constexpr int h2dMemcpySleepMs = 100;

    CudaCoreMatMatMulRunner cudaCoreMatMatMulRunner(m, n, k);
    std::future<std::vector<float>> cudaCoreMatMatMulFuture
        = std::async(std::launch::async, runTask, std::ref(cudaCoreMatMatMulRunner), cudaCoreMatMatMulSleepMs, expDurationSec);

    TensorCoreMatMatMulRunner tensorCoreMatMatMulRunner(m, n, k);
    std::future<std::vector<float>> tensorCoreMatMatMulFuture = std::async(
        std::launch::async, runTask, std::ref(tensorCoreMatMatMulRunner), tensorCoreMatMatMulSleepMs, expDurationSec);

    H2DMemcpyRunner h2dMemcpyRunner(m, n, k);
    std::future<std::vector<float>> h2dMemcpyFuture
        = std::async(std::launch::async, runTask, std::ref(h2dMemcpyRunner), h2dMemcpySleepMs, expDurationSec);

    printTimeRecords(cudaCoreMatMatMulFuture.get());
    printTimeRecords(tensorCoreMatMatMulFuture.get());
    printTimeRecords(h2dMemcpyFuture.get());

    return 0;
}