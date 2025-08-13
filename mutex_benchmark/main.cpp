#include <iostream>
#include <thread>
#include <vector>
#include <future>
#include <random>
#include <algorithm>

using namespace std;

#include "timer.h"

class LatencyRecorderWithMutex
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

class LatencyRecorderWithoutMutex
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

int main()
{
    // -------------
    // Configuration
    const int kMaxNumRecords = 1000000;
    const int kNumThreads = 10;
    const int kSleepDurationMs = 1;
    const int kNumRecordsPerThread = 10000;

    cout << "Configuration:" << endl;
    cout << "Max number of records: "
    // --------------
    // Validate configuration
    if (kNumThreads * kNumRecordsPerThread > kMaxNumRecords)
    {
        cerr << "Error: Total records exceed maximum allowed records." << endl;
        return 1;
    }

    // --------------------
    // Run recorder with mutex
    {
        LatencyRecorderWithMutex recorderWithMutex(kMaxNumRecords);
        Timer timer;
        timer.tic();

        // --------------
        // Create threads to record latencies
        vector<thread> threads;
        for (int threadIdx = 0; threadIdx < kNumThreads; threadIdx++)
        {
            threads.emplace_back([&recorderWithMutex, kSleepDurationMs]()
                                 {
                                 for (int recordIdx = 0; recordIdx < kNumRecordsPerThread; recordIdx++)
                                 {
                                    if (kSleepDurationMs > 0)
                                    {
                                        std::this_thread::sleep_for(std::chrono::milliseconds(kSleepDurationMs));
                                    }
                                    float latency = recordIdx;
                                    recorderWithMutex.record(latency);
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
        float totalTimeMs = timer.tocMs();
        float totalSleepTimeMs = kNumRecordsPerThread * kSleepDurationMs;
        float totalRecordTimeMs = totalTimeMs - totalSleepTimeMs;
        cout << "Total time with mutex: " << totalTimeMs << " ms" << endl;
        cout << "Total record time with mutex: " << totalRecordTimeMs << " ms" << endl;
        cout << "Average record time with mutex: " << totalRecordTimeMs / kNumRecordsPerThread << " ms" << endl;
    }

    // ---------------------
    // Run recorder without mutex
    {
        LatencyRecorderWithoutMutex recorderWithoutMutex(kMaxNumRecords);
        Timer timer;
        timer.tic();

        // --------------
        // Create threads to record latencies
        vector<thread> threads;
        for (int threadIdx = 0; threadIdx < kNumThreads; threadIdx++)
        {
            threads.emplace_back([&recorderWithoutMutex, kSleepDurationMs]()
                                 {
                                 for (int recordIdx = 0; recordIdx < kNumRecordsPerThread; recordIdx++)
                                 {
                                    if (kSleepDurationMs > 0)
                                    {
                                        std::this_thread::sleep_for(std::chrono::milliseconds(kSleepDurationMs));
                                    }
                                    float latency = recordIdx;
                                    recorderWithoutMutex.record(latency);
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
        float totalTimeMs = timer.tocMs();
        float totalSleepTimeMs = kNumRecordsPerThread * kSleepDurationMs;
        float totalRecordTimeMs = totalTimeMs - totalSleepTimeMs;
        cout << "Total time without mutex: " << totalTimeMs << " ms" << endl;
        cout << "Total record time without mutex: " << totalRecordTimeMs << " ms" << endl;
        cout << "Average record time without mutex: " << totalRecordTimeMs / kNumRecordsPerThread << " ms" << endl;
    }

    return 0;
}