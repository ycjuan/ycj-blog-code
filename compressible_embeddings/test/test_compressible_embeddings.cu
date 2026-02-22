#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>
#include <tuple>
#include <unordered_set>

#include "manager/emb_dataset_manager.hpp"
#include "resident/resident_partition_config.hpp"
#include "utils/util.hpp"
#include "working/working_emb_dataset.hpp"

constexpr size_t kNumDocs = 50000;
constexpr size_t kTotalEmbDim = 128;
constexpr size_t kMaxWorkingSetSize = 50000;
constexpr size_t kNumDocsToDensify = 10000;
constexpr size_t kDensifiedEmbIdxBeginIncl = 3;
constexpr size_t kDensifiedEmbIdxEndExcl = 125;
constexpr size_t kNumCentroids = 16;
constexpr size_t kNumBitsPerDim = 2;
const std::vector<ResidentPartitionConfig> kResidentPartitionConfigs
    = { ResidentPartitionConfig(0, 48, MemLayout::ROW_MAJOR), ResidentPartitionConfig(64, 96, MemLayout::ROW_MAJOR) };
constexpr float kCentroidStdDev = 1.0f;
constexpr float kCentroidMean = 0.0f;
constexpr float kCacheRate = 0.9f;
constexpr int kNumDensifyTrials = 20;

// Generate random document data given the centroids and std devs.
std::tuple<std::vector<T_DOC_IDX>, std::vector<std::vector<T_EMB>>, std::vector<int>> genRandDocData(
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

// Generate the next docIdxList to densify based on the current docIdxList and the cache rate.
// If the cache rate is 0.9, then 90% of the current docIdxList will be kept, while the remaining 10% will be randomly generated.
std::pair<std::vector<T_DOC_IDX>, float> genNextDocIdxList(const std::vector<T_DOC_IDX>& current, int trial)
{
    static std::default_random_engine generator(trial);
    size_t numToKeep = static_cast<size_t>(kNumDocsToDensify * kCacheRate);
    std::uniform_int_distribution<T_DOC_IDX> dist(0, kNumDocs - 1);

    std::unordered_set<T_DOC_IDX> nextSet(current.begin(), current.begin() + numToKeep);
    while (nextSet.size() < kNumDocsToDensify)
    {
        nextSet.insert(dist(generator));
    }

    std::unordered_set<T_DOC_IDX> prevSet(current.begin(), current.end());
    size_t overlapCount = 0;
    for (T_DOC_IDX idx : nextSet)
    {
        if (prevSet.count(idx))
        {
            overlapCount++;
        }
    }
    assert(overlapCount >= numToKeep && "Cache rate verification failed: fewer overlapping docs than expected");

    float cacheRate = static_cast<float>(overlapCount) / kNumDocsToDensify;
    std::vector<T_DOC_IDX> next(nextSet.begin(), nextSet.end());
    std::sort(next.begin(), next.end());
    return { next, cacheRate };
}

// Generate a random docIdxList to densify.
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

// Generate random centroids and std devs.
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
    for (const auto& config : kResidentPartitionConfigs)
    {
        if (embIdx >= config.getEmbDimBeginIncl() && embIdx < config.getEmbDimEndExcl())
        {
            return false;
        }
    }
    return true;
}

// Verify the densification result.
float verifyDensification(const WorkingEmbDataset& workingEmbDataset,
                          const std::vector<T_DOC_IDX>& docIdxList,
                          const std::vector<std::vector<T_EMB>>& emb2D)
{
    std::vector<T_EMB> v_workingEmbDataset(kMaxWorkingSetSize * kTotalEmbDim);
    CHECK_CUDA(cudaMemcpy(v_workingEmbDataset.data(),
                          workingEmbDataset.data(),
                          kNumDocsToDensify * (kDensifiedEmbIdxEndExcl - kDensifiedEmbIdxBeginIncl) * sizeof(T_EMB),
                          cudaMemcpyDeviceToHost));

    float compressedErrorSum = 0.0f;
    size_t compressedCount = 0;

    for (size_t docIdx = 0; docIdx < docIdxList.size(); ++docIdx)
    {
        for (size_t embIdx = kDensifiedEmbIdxBeginIncl; embIdx < kDensifiedEmbIdxEndExcl; ++embIdx)
        {
            size_t memAddr = getMemAddrRowMajor(docIdx,
                                                embIdx - kDensifiedEmbIdxBeginIncl,
                                                kNumDocsToDensify,
                                                kDensifiedEmbIdxEndExcl - kDensifiedEmbIdxBeginIncl);
            float val = static_cast<float>(v_workingEmbDataset.at(memAddr));
            float ref = static_cast<float>(emb2D.at(docIdxList.at(docIdx)).at(embIdx));

            if (isCompressedDim(embIdx))
            {
                float error = std::abs(val - ref);
                compressedErrorSum += error;
                compressedCount++;
                if (error > 1.1f * kCentroidStdDev)
                {
                    std::ostringstream oss;
                    oss << "Compressed dim: docIdx = " << docIdx << ", embIdx = " << embIdx << ", val(" << val
                        << ") != ref(" << ref << ")"
                        << ", error = " << error << ", threshold = " << kCentroidStdDev;
                    throw std::runtime_error(oss.str());
                }
            }
            else
            {
                if (val != ref)
                {
                    std::ostringstream oss;
                    oss << "Resident dim: docIdx = " << docIdx << ", embIdx = " << embIdx << ", val(" << val
                        << ") != ref(" << ref << ")"
                        << ", memAddr: " << memAddr;
                    throw std::runtime_error(oss.str());
                }
            }
        }
    }

    return compressedErrorSum / compressedCount;
}

int main()
{
    if (!hasCudaDevice())
    {
        return 0;
    }

    // ------------------
    // Prepare the data

    // --------
    // Generate the centroids
    auto [centroidEmbs, centroidStdDevs] = genRandCentroids();

    // --------
    // Generate 2D embeddings and centroid indices
    auto [docIdxList, emb2D, centroidIdxList] = genRandDocData(centroidEmbs, centroidStdDevs);

    // ------------------
    // Initialize the EmbDatasetManager
    EmbDatasetManager embDatasetManager(kNumDocs,
                                        kTotalEmbDim,
                                        kResidentPartitionConfigs,
                                        kMaxWorkingSetSize,
                                        kNumBitsPerDim,
                                        centroidEmbs,
                                        centroidStdDevs);
    // --------
    // Ingest the data into the EmbDatasetManager
    embDatasetManager.update(docIdxList, emb2D);

    // --------
    // Generate random docs to densify
    std::vector<T_DOC_IDX> docIdxListToDensify = genRandomDocIdxList();

    // --------
    // Some statistics
    float totalDensifyTimeMs = 0.0f;
    float totalCompressedError = 0.0f;
    float totalCacheRate = 0.0f;

    // --------
    // Densify the data
    for (int trial = -3; trial < kNumDensifyTrials; ++trial)
    {
        // --------
        // Start the formal benchmark after 3 warmup trials
        if (trial == 0)
        {
            embDatasetManager.getLastTimeRecordAndReset();
            totalDensifyTimeMs = 0.0f;
            totalCompressedError = 0.0f;
            totalCacheRate = 0.0f;
        }

        // --------
        // Densification
        Timer timer;
        timer.tic();
        const WorkingEmbDataset& workingEmbDataset = embDatasetManager.densify(docIdxListToDensify,
                                                                               kDensifiedEmbIdxBeginIncl,
                                                                               kDensifiedEmbIdxEndExcl,
                                                                               MemLayout::ROW_MAJOR);
        totalDensifyTimeMs += timer.tocMs();

        // --------
        // Verify the result
        totalCompressedError += verifyDensification(workingEmbDataset, docIdxListToDensify, emb2D);

        // --------
        // Generate the next docIdxList to densify
        auto [nextList, cacheRate] = genNextDocIdxList(docIdxListToDensify, trial);
        totalCacheRate += cacheRate;
        docIdxListToDensify = std::move(nextList);
    }

    // --------
    // Print the statistics
    std::cout << std::fixed << std::setprecision(3)
              << "\n===== Summary (avg over " << kNumDensifyTrials << " trials) =====\n"
              << "Densification time: " << totalDensifyTimeMs / kNumDensifyTrials << " ms\n"
              << "Cache rate: " << totalCacheRate / kNumDensifyTrials << "\n";
    std::cout << std::setprecision(6) << "Compressed dim avg error: " << totalCompressedError / kNumDensifyTrials << "\n";
    std::cout << "\n===== Time Record =====\n";
    embDatasetManager.getLastTimeRecordAndReset().print();

    return 0;
}