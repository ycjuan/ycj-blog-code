#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <chrono>
#include <numeric>

using namespace std;

#include "timer.h"

// -------------
// Configuration
constexpr int kMaxNumRecords = 1000000;
constexpr int kNumThreads = 64;
constexpr int kSleepDurationMs = 0;
constexpr int kNumRecordsPerThread = 10000;

class LatencyRecorderBase
{
public:
    virtual void record(float latencyMs) = 0;
    virtual ~LatencyRecorderBase() = default;
};

class LatencyRecorderWithMutex : public LatencyRecorderBase
{
public:
    LatencyRecorderWithMutex(int maxNumRecords)
    {
        latencies_.reserve(maxNumRecords);
    }

    void record(float latencyMs)
    {
        lock_guard<mutex> lock(mutex_);
        latencies_.push_back(latencyMs);
    }

private:
    mutex mutex_;
    vector<float> latencies_;
};

class LatencyRecorderWithoutMutex : public LatencyRecorderBase
{
public:
    LatencyRecorderWithoutMutex(int maxNumRecords)
    {
        latencies_.resize(maxNumRecords);
    }
    void record(float latencyMs)
    {
        latencies_[currIdx_++] = latencyMs;
    }

private:
    atomic_int currIdx_{0};
    vector<float> latencies_;
};

void runExperiment(LatencyRecorderBase &recorder)
{
        // --------------
        // Create threads to record latencies
        vector<thread> threads;
        vector<int64_t> recordTimeMicrosecondSum(kNumThreads, 0);
        for (int threadIdx = 0; threadIdx < kNumThreads; threadIdx++)
        {
            threads.emplace_back([&recorder, &recordTimeMicrosecondSum, threadIdx]()
                                 {
                                 for (int recordIdx = 0; recordIdx < kNumRecordsPerThread; recordIdx++)
                                 {
                                    if (kSleepDurationMs > 0)
                                    {
                                        std::this_thread::sleep_for(std::chrono::milliseconds(kSleepDurationMs));
                                    }
                                    float dummyLatency = recordIdx; // The value of latency doesn't really matter for this benchmark

                                    Timer timer;
                                    timer.tic();
                                    recorder.record(dummyLatency);
                                    int64_t recordTimeMicrosec = timer.tocMicrosec();
                                    recordTimeMicrosecondSum[threadIdx] += recordTimeMicrosec;
                                 } });
        }

        // --------------
        // Wait for all threads to finish
        for (auto &t : threads)
        {
            t.join();
        }

        // --------------
        // Calculate total time taken
        int64_t recordTimeMicrosecondTotal = accumulate(recordTimeMicrosecondSum.begin(), recordTimeMicrosecondSum.end(), 0);
        cout << "Total record time: " << recordTimeMicrosecondTotal << " microseconds" << endl;
        cout << "Average record time per transaction: " << (double)recordTimeMicrosecondTotal / (kNumThreads * kNumRecordsPerThread) << " microseconds" << endl;
}

int main()
{
    cout << "Configuration:" << endl;
    cout << "Max number of records: " << kMaxNumRecords << endl;
    cout << "Number of threads: " << kNumThreads << endl;
    cout << "Sleep duration (ms): " << kSleepDurationMs << endl;
    cout << "Number of records per thread: " << kNumRecordsPerThread << endl;
    cout << endl;

    // --------------
    // Validate configuration
    if (kNumThreads * kNumRecordsPerThread > kMaxNumRecords)
    {
        cerr << "Error: Total records exceed maximum allowed records." << endl;
        return 1;
    }

    // --------------------
    // Run recorder with mutex
    cout << "Running recorder with mutex..." << endl;
    cout << "--------------------------------" << endl;
    LatencyRecorderWithMutex recorderWithMutex(kMaxNumRecords);
    runExperiment(recorderWithMutex);

    // ---------------------
    // Run recorder without mutex
    cout << endl << endl;
    cout << "Running recorder without mutex..." << endl;
    cout << "-----------------------------------" << endl;
    LatencyRecorderWithoutMutex recorderWithoutMutex(kMaxNumRecords);
    runExperiment(recorderWithoutMutex);

    return 0;
}