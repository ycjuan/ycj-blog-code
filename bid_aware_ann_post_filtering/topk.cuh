#ifndef TOPK_CUH
#define TOPK_CUH

#include <vector>

#include "data_struct.cuh"

struct TopkRetrievalParam
{
    ReqDocPair *d_pair;
    ReqDocPair *d_buffer;
    int numReqDocPairs;
    int numToRetrieve;
    int numRetrieved;
    float timeMsUpdateCounter = 0;
    float timeMsCopyCounterToCpu = 0;
    float timeMsFindLowestBucket = 0;
    float timeMsPrefilter = 0;
    float timeMsSort = 0;
    float timeMsCopyBackToCpu = 0;
    float timeMsTotal = 0;
};

class Topk
{
public:
    void init(float minScore, float maxScore);

    void reset();

    std::vector<ReqDocPair> retrieveTopk(TopkRetrievalParam &param);

    void retrieveTopkApprox(TopkRetrievalParam &param);

    __device__ __host__ int getCounterIdx(int slot, int bucket)
    {
        return slot * kNumBuckets_ + bucket;
    }

    __device__ int bucketize(float score)
    {
        score = min(maxScore_, max(minScore_, score));
        score = (score - minScore_) / (maxScore_ - minScore_);
        return (int)(score * kGranularity_);
    }

    __device__ void updateCounter(ReqDocPair rd)
    {
        int slot = rd.docIdx % kNumSlots_;
        int bucket = bucketize(rd.score);
        int counterIdx = getCounterIdx(slot, bucket);

        atomicAdd(&d_counter_[counterIdx], 1);
    }

    __device__ bool operator()(const ReqDocPair &rd)
    {
        int bucket = bucketize(rd.score);
        return bucket >= lowestBucket_;
    }

private:
    float minScore_ = -1;
    float maxScore_ = 1;
    const int kGranularity_ = 512;
    const int kNumBuckets_ = kGranularity_ + 1;
    // For example, if you have kMinScore = -1.0, kMaxScore = 1.0, kGranularity = 2,
    // then you will have 3 buckets: [-1, 0, 1]
    const int kNumSlots_ = 16;
    // Each bucket has 16 slots. A GPU thread will randomly write into one of the thread,
    // and then the count will finally be accumulated to the first slot.
    // The purpose of doing this is to minimize too much concurrent write to the same memory location with atomicAdd.
    const int kSize_d_counter_ = kNumBuckets_ * kNumSlots_;
    const int kSize_byte_d_counter_ = kSize_d_counter_ * sizeof(int);

    int *d_counter_ = nullptr;
    int *h_counter_ = nullptr;
    int lowestBucket_ = 0;

    void findLowestBucket(std::vector<int> &v_counter, int numToRetrieve, int &lowestBucket, int &numReqDocPairsGreaterThanLowestBucket);
};

#endif // TOPK_CUH