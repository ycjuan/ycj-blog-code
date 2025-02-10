#include <iostream>
#include <thread>
#include <vector>
#include <future>
#include <random>
#include <algorithm>

#include "retriever.h"
#include "dispatcher.h"
#include "config.h"

using namespace std;

void analyzeResponses(const vector<Response> &responses)
{
    vector<float> latencies;
    for (const Response &response : responses)
    {
        latencies.push_back(response.latencyMs);
    }
    sort(latencies.begin(), latencies.end());
    float p50Latency = latencies[static_cast<int>(latencies.size() * 0.50)];
    float p90Latency = latencies[static_cast<int>(latencies.size() * 0.90)];
    float p95Latency = latencies[static_cast<int>(latencies.size() * 0.95)];
    float p99Latency = latencies[static_cast<int>(latencies.size() * 0.99)];
    float p100Latency = latencies.back();
    cout << "P50 latency: " << p50Latency << " ms" << endl;
    cout << "P90 latency: " << p90Latency << " ms" << endl;
    cout << "P95 latency: " << p95Latency << " ms" << endl;
    cout << "P99 latency: " << p99Latency << " ms" << endl;
    cout << "P100 latency: " << p100Latency << " ms" << endl;

    vector<int> batchSizes;
    for (const Response &response : responses)
    {
        batchSizes.push_back(response.batchSize);
    }
    sort(batchSizes.begin(), batchSizes.end());
    int p50BatchSize = batchSizes[static_cast<int>(batchSizes.size() * 0.50)];
    int p90BatchSize = batchSizes[static_cast<int>(batchSizes.size() * 0.90)];
    int p95BatchSize = batchSizes[static_cast<int>(batchSizes.size() * 0.95)];
    int p99BatchSize = batchSizes[static_cast<int>(batchSizes.size() * 0.99)];
    int p100BatchSize = batchSizes.back();
    cout << "P50 batch size: " << p50BatchSize << endl;
    cout << "P90 batch size: " << p90BatchSize << endl;
    cout << "P95 batch size: " << p95BatchSize << endl;
    cout << "P99 batch size: " << p99BatchSize << endl;
    cout << "P100 batch size: " << p100BatchSize << endl;
}

int main()
{
    cout << "QPS: " << kQPS << ", ExpTimeSec: " << kExpTimeSec << ", NumReqs: " << kNumReqs << endl;
    
    Retriever retriever(kLatencyMsMap, kBatchLatencyMultiplierMap);
    Dispatcher dispatcher(retriever);

    default_random_engine generator;
    uniform_int_distribution<int> distEmbVer(0, 99);
    vector<future<Response>> responseFutures;

    thread processThread([&dispatcher] { dispatcher.processRequests(); });

    for (int i = 0; i < kNumReqs; i++)
    {
        Request request;
        request.requestId = i;
        request.creationTimepoint = chrono::high_resolution_clock::now();
        request.timeoutTimepoint = request.creationTimepoint + chrono::seconds(1);
        request.groupName = getGroupName(distEmbVer(generator));

        responseFutures.push_back(async(launch::async, [&dispatcher, request] { return dispatcher.retrieve(request); }));
        this_thread::sleep_for(chrono::milliseconds(1000 / kQPS));
    }

    vector<Response> responses;
    for (int i = 0; i < kNumReqs; i++)
    {
        responses.push_back(responseFutures[i].get());
    }

    dispatcher.stop();

    processThread.join();

    analyzeResponses(responses);

    return 0;
}