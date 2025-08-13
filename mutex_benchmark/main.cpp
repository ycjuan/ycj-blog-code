#include <iostream>
#include <thread>
#include <vector>
#include <future>
#include <random>
#include <algorithm>

using namespace std;

#include "timer.h"

// -------------
// Configuration
constexpr int kMaxNumRecords = 1000000;
constexpr int kNumThreads = 48;
constexpr int kSleepDurationMs = 10;
constexpr int kNumRecordsPerThread = 1000;

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
        Timer timer;
        timer.tic();

        // --------------
        // Create threads to record latencies
        vector<thread> threads;
        for (int threadIdx = 0; threadIdx < kNumThreads; threadIdx++)
        {
            threads.emplace_back([&recorder]()
                                 {
                                 for (int recordIdx = 0; recordIdx < kNumRecordsPerThread; recordIdx++)
                                 {
                                    if (kSleepDurationMs > 0)
                                    {
                                        std::this_thread::sleep_for(std::chrono::milliseconds(kSleepDurationMs));
                                    }
                                    float latency = recordIdx;
                                    recorder.record(latency);
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
        int64_t totalTimeMicrosec = timer.tocMicrosec();
        int64_t totalSleepTimeMicrosec = kNumRecordsPerThread * kSleepDurationMs * 1000;
        int64_t totalRecordTimeMicrosec = totalTimeMicrosec - totalSleepTimeMicrosec;
        cout << "Total time: " << totalTimeMicrosec << " microseconds" << endl;
        cout << "Total sleep time: " << totalSleepTimeMicrosec << " microseconds" << endl;
        cout << "Total record time: " << totalRecordTimeMicrosec << " microseconds" << endl;
        cout << "Average record time: " << (double)totalRecordTimeMicrosec / kNumRecordsPerThread << " microseconds" << endl;

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
