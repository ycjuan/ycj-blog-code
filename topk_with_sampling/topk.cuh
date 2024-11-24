#ifndef TOPK_CUH
#define TOPK_CUH

#include <vector>

#include "common.cuh"

class TopkBucketSort
{
public:
    void init();

    void reset();

    std::vector<Pair> retrieveTopk(Pair *d_doc, Pair *d_buffer, int numDocs, int numToRetrieve, float &timeMs);

    __device__ __host__ int getCounterIdx(int slot, int bucket)
    {
        return slot * kNumBuckets_ + bucket;
    }

    __device__ int bucketize(float score)
    {
        score = min(kMaxScore_, max(kMinScore_, score));
        score = (score - kMinScore_) / (kMaxScore_ - kMinScore_);
        return (int)(score * kGranularity_);
    }

    __device__ void updateCounter(Pair doc)
    {
        int slot = doc.docId % kNumSlots_;
        int bucket = bucketize(doc.score);
        int counterIdx = getCounterIdx(slot, bucket);

        atomicAdd(&d_counter_[counterIdx], 1);
    }

    __device__ bool operator()(const Pair &doc)
    {
        int bucket = bucketize(doc.score);
        return bucket >= lowestBucket_;
    }

private:
    const float kMinScore_ = -1;
    const float kMaxScore_ = 1;
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

    void findLowestBucket(std::vector<int> &v_counter, int numToRetrieve, int &lowestBucket, int &numDocsGreaterThanLowestBucket);
};

std::vector<Pair> retrieveTopkGpuFullSort(Pair *d_doc, int numDocs, int numToRetrieve, float &timeMs);
std::vector<Pair> retrieveTopkCpuFullSort(std::vector<Pair> &v_doc, int numToRetrieve, float &timeMs);

#endif // TOPK_CUH