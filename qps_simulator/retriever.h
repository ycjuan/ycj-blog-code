#ifndef RETRIEVER_H
#define RETRIEVER_H

#include <string>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <vector>

using namespace std;

struct Request
{
    size_t requestId;
    chrono::high_resolution_clock::time_point creationTimepoint;
    chrono::high_resolution_clock::time_point timeoutTimepoint;
    string groupName;
};

struct Response
{
    float latencyMs;
    int batchSize;
};

class Retriever
{
public:
    Retriever(unordered_map<string, float> latencyMsMap,
              unordered_map<int, float> batchLatencyMultiplierMap_)
        : latencyMsMap_(latencyMsMap), batchLatencyMultiplierMap_(batchLatencyMultiplierMap_) {}

    Response retrieve(const vector<Request>& requests) const
    {
        // ----------------------
        // preparation
        if (requests.empty())
        {
            return Response();
        }
        const Request& request0 = requests[0];
        const string groupName = request0.groupName;
        const int batchSize = requests.size();
        for (int i = 1 ; i < batchSize; i++)
        {
            if (requests[i].groupName != groupName)
            {
                throw runtime_error("All requests must have the same group name");
            }
        }

        // ----------------------
        // simulate the latency
        float latencyMs = latencyMsMap_.at(groupName);
        latencyMs *= batchLatencyMultiplierMap_.at(batchSize);
        this_thread::sleep_for(chrono::microseconds(int(latencyMs * 1000)));

        // ----------------------
        // In our simulator we just return empty object. In reality, engineers would return the actual retrieved results for each request.
        return Response();
    }

    int getMaxBatchSize() const { return maxBatchSize_; }

private:
    const int maxBatchSize_ = 8;
    const unordered_map<string, float> latencyMsMap_;
    const unordered_map<int, float> batchLatencyMultiplierMap_;
};

#endif