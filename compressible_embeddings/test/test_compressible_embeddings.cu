#include <cassert>
#include <random>
#include <iostream>
#include <unordered_set>
#include <algorithm>
#include <tuple>

#include "manager/emb_index_manager.hpp"
#include "resident/resident_partition_config.hpp"
#include "working/working_emb_index.hpp"
#include "utils/util.hpp"

constexpr size_t kNumDocs = 10000;
constexpr size_t kTotalEmbDim = 128;
constexpr size_t kMaxWorkingSetSize = 1000;
constexpr size_t kNumDocsToDensify = 500;
constexpr size_t kDensifiedEmbIdxBeginIncl = 3;
constexpr size_t kDensifiedEmbIdxEndExcl = 125;
constexpr size_t kNumCentroids = 16;
constexpr size_t kNumBitsPerDim = 2;
const std::vector<ResidentPartitionConfig> kResidentPartitionConfigs
    = { ResidentPartitionConfig(0, 48, MemLayout::ROW_MAJOR), ResidentPartitionConfig(64, 96, MemLayout::ROW_MAJOR) };
constexpr float kCentroidStdDev = 1.0f;
constexpr float kCentroidMean = 0.0f;
constexpr float kCacheRate = 0.5f;
constexpr size_t kNumDensifyTrials = 20;

std::tuple<std::vector<T_DOC_IDX>, std::vector<std::vector<T_EMB>>, std::vector<int>> populateRandomEmbIndex(
    const std::vector<std::vector<float>>& centroidEmbs,
    const std::vector<std::vector<float>>& centroidStdDevs)
{
    size_t numCentroids = centroidEmbs.size();
    std::vector<T_DOC_IDX> docIdxList(kNumDocs);
    std::vector<std::vector<T_EMB>> emb2D(kNumDocs);
    std::vector<int> centroidIdxList(kNumDocs);
    std::default_random_engine generator;
    for (size_t docIdx = 0; docIdx < kNumDocs; ++docIdx)
    {
        docIdxList.at(docIdx) = docIdx;
        emb2D.at(docIdx).resize(kTotalEmbDim);
        int centroidIdx = docIdx % numCentroids;
        centroidIdxList.at(docIdx) = centroidIdx;
        for (size_t embIdx = 0; embIdx < kTotalEmbDim; ++embIdx)
        {
            float centroid = centroidEmbs[centroidIdx][embIdx];
            float stdDev = centroidStdDevs[centroidIdx][embIdx];
            std::normal_distribution<float> distribution(centroid, stdDev);
            float randVal = distribution(generator);
            randVal = std::min(randVal, centroid + 3 * stdDev);
            randVal = std::max(randVal, centroid - 3 * stdDev);
            emb2D.at(docIdx).at(embIdx) = (T_EMB)randVal;
        }
    }
    return std::make_tuple(docIdxList, emb2D, centroidIdxList);
}

std::vector<T_DOC_IDX> genRandomDocIdxList()
{
    std::unordered_set<T_DOC_IDX> docIdxSet;
    std::default_random_engine generator;
    std::uniform_int_distribution<T_DOC_IDX> distribution(0, kNumDocs - 1);
    while (docIdxSet.size() < kNumDocsToDensify)
    {
        docIdxSet.insert(distribution(generator));
    }
    std::vector<T_DOC_IDX> docIdxList(docIdxSet.begin(), docIdxSet.end());
    std::sort(docIdxList.begin(), docIdxList.end());
    return docIdxList;
}

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> genRandCentroids()
{
    std::default_random_engine generator(42);
    std::normal_distribution<float> distribution(kCentroidMean, kCentroidStdDev);
    std::vector<std::vector<float>> centroidEmbs(kNumCentroids, std::vector<float>(kTotalEmbDim));
    std::vector<std::vector<float>> centroidStdDevs(kNumCentroids, std::vector<float>(kTotalEmbDim, kCentroidStdDev));
    for (size_t c = 0; c < kNumCentroids; c++)
    {
        for (size_t d = 0; d < kTotalEmbDim; d++)
        {
            centroidEmbs[c][d] = std::min(100.0f, std::max(-100.0f, distribution(generator)));
        }
    }
    return std::make_pair(centroidEmbs, centroidStdDevs);
}

bool isCompressedDim(size_t embIdx)
{
    return (embIdx >= 48 && embIdx < 64) || embIdx >= 96;
}

void verifyDensification(const WorkingEmbDataset& workingEmbIndex,
                         const std::vector<T_DOC_IDX>& docIdxList,
                         const std::vector<std::vector<T_EMB>>& emb2D)
{
    std::vector<T_EMB> v_workingEmbIndex(kMaxWorkingSetSize * kTotalEmbDim);
    CHECK_CUDA(cudaMemcpy(v_workingEmbIndex.data(),
                          workingEmbIndex.data(),
                          kNumDocsToDensify * (kDensifiedEmbIdxEndExcl - kDensifiedEmbIdxBeginIncl) * sizeof(T_EMB),
                          cudaMemcpyDeviceToHost));

    float compressedErrorSum = 0.0f;
    size_t compressedCount = 0;

    for (size_t docIdx = 0; docIdx < docIdxList.size(); ++docIdx)
    {
        for (size_t embIdx = kDensifiedEmbIdxBeginIncl; embIdx < kDensifiedEmbIdxEndExcl; ++embIdx)
        {
            size_t memAddr = getMemAddrRowMajor(docIdx, embIdx - kDensifiedEmbIdxBeginIncl, kNumDocsToDensify, kDensifiedEmbIdxEndExcl - kDensifiedEmbIdxBeginIncl);
            float val = static_cast<float>(v_workingEmbIndex.at(memAddr));
            float ref = static_cast<float>(emb2D.at(docIdxList.at(docIdx)).at(embIdx));

            if (isCompressedDim(embIdx))
            {
                float error = std::abs(val - ref);
                compressedErrorSum += error;
                compressedCount++;
                if (error > 1.1f * kCentroidStdDev)
                {
                    std::ostringstream oss;
                    oss << "Compressed dim: docIdx = " << docIdx << ", embIdx = " << embIdx
                        << ", val(" << val << ") != ref(" << ref << ")"
                        << ", error = " << error << ", threshold = " << kCentroidStdDev;
                    throw std::runtime_error(oss.str());
                }
            }
            else
            {
                if (val != ref)
                {
                    std::ostringstream oss;
                    oss << "Resident dim: docIdx = " << docIdx << ", embIdx = " << embIdx
                        << ", val(" << val << ") != ref(" << ref << ")"
                        << ", memAddr: " << memAddr;
                    throw std::runtime_error(oss.str());
                }
            }
        }
    }

    printf("Compressed dim avg error: %.6f (over %zu samples)\n", compressedErrorSum / compressedCount, compressedCount);
    printf("!!!!!!!!!!!! Densification verified successfully !!!!!!!!!!!!\n");
}

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0)
    {
        std::cout << "!!!!!!!!!! No CUDA device found or error occurred: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }

    auto [centroidEmbs, centroidStdDevs] = genRandCentroids();

    EmbIndexManager embIndexManager(kNumDocs, kTotalEmbDim, kResidentPartitionConfigs, kMaxWorkingSetSize,
                                    kNumBitsPerDim, centroidEmbs, centroidStdDevs);

    auto [docIdxList, emb2D, centroidIdxList] = populateRandomEmbIndex(centroidEmbs, centroidStdDevs);
    embIndexManager.update(docIdxList, emb2D);

    std::default_random_engine trialGenerator(123);
    std::uniform_int_distribution<T_DOC_IDX> docIdxDist(0, kNumDocs - 1);

    std::vector<T_DOC_IDX> docIdxListToDensify = genRandomDocIdxList();

    for (size_t trial = 0; trial < kNumDensifyTrials; ++trial)
    {
        std::cout << "===== Trial " << trial << " =====" << std::endl;

        const WorkingEmbDataset& workingEmbIndex = embIndexManager.densify(docIdxListToDensify,
                                                                         kDensifiedEmbIdxBeginIncl,
                                                                         kDensifiedEmbIdxEndExcl,
                                                                         MemLayout::ROW_MAJOR);

        verifyDensification(workingEmbIndex, docIdxListToDensify, emb2D);

        // Build next trial's docIdxList: kCacheRate from current, (1 - kCacheRate) new random.
        size_t numToKeep = static_cast<size_t>(kNumDocsToDensify * kCacheRate);

        std::unordered_set<T_DOC_IDX> nextSet(docIdxListToDensify.begin(),
                                               docIdxListToDensify.begin() + numToKeep);
        while (nextSet.size() < kNumDocsToDensify)
        {
            T_DOC_IDX candidate = docIdxDist(trialGenerator);
            nextSet.insert(candidate);
        }
        // Verify cache rate: count overlap between current docIdxListToDensify and nextSet.
        std::unordered_set<T_DOC_IDX> prevSet(docIdxListToDensify.begin(), docIdxListToDensify.end());
        size_t overlapCount = 0;
        for (T_DOC_IDX idx : nextSet)
        {
            if (prevSet.count(idx)) overlapCount++;
        }
        float actualCacheRate = static_cast<float>(overlapCount) / kNumDocsToDensify;
        printf("Trial %zu cache rate: %.3f (expected >= %.3f, overlap %zu / %zu)\n",
               trial, actualCacheRate, kCacheRate, overlapCount, kNumDocsToDensify);
        assert(overlapCount >= numToKeep && "Cache rate verification failed: fewer overlapping docs than expected");

        docIdxListToDensify.assign(nextSet.begin(), nextSet.end());
        std::sort(docIdxListToDensify.begin(), docIdxListToDensify.end());
    }

    return 0;
}