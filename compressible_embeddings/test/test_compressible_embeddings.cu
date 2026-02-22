#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <unordered_set>

#include "manager/emb_dataset_manager.hpp"
#include "resident/resident_partition_config.hpp"
#include "utils/util.hpp"
#include "working/working_emb_dataset.hpp"

struct ExpSetting
{
    size_t numDocs = 50000;
    size_t totalEmbDim = 128;
    size_t maxWorkingSetSize = 50000;
    size_t numDocsToDensify = 10000;
    size_t densifiedEmbIdxBeginIncl = 3;
    size_t densifiedEmbIdxEndExcl = 125;
    size_t numCentroids = 16;
    size_t numBitsPerDim = 2;
    std::vector<ResidentPartitionConfig> residentPartitionConfigs
        = { ResidentPartitionConfig(0, 48, MemLayout::ROW_MAJOR),
            ResidentPartitionConfig(64, 96, MemLayout::ROW_MAJOR) };
    float centroidStdDev = 1.0f;
    float centroidMean = 0.0f;
    float cacheRate = 0.9f;
    int numDensifyTrials = 20;

    void print() const
    {
        std::cout << "===== ExpSetting =====\n"
                  << "numDocs: " << numDocs << "\n"
                  << "totalEmbDim: " << totalEmbDim << "\n"
                  << "maxWorkingSetSize: " << maxWorkingSetSize << "\n"
                  << "numDocsToDensify: " << numDocsToDensify << "\n"
                  << "densifiedEmbIdxBeginIncl: " << densifiedEmbIdxBeginIncl << "\n"
                  << "densifiedEmbIdxEndExcl: " << densifiedEmbIdxEndExcl << "\n"
                  << "numCentroids: " << numCentroids << "\n"
                  << "numBitsPerDim: " << numBitsPerDim << "\n"
                  << "residentPartitionConfigs: [";
        for (size_t i = 0; i < residentPartitionConfigs.size(); ++i)
        {
            if (i > 0)
            {
                std::cout << ", ";
            }
            std::cout << "[" << residentPartitionConfigs.at(i).getEmbDimBeginIncl() << ", "
                      << residentPartitionConfigs.at(i).getEmbDimEndExcl() << ")";
        }
        std::cout << "]\n"
                  << "centroidStdDev: " << centroidStdDev << "\n"
                  << "centroidMean: " << centroidMean << "\n"
                  << "cacheRate: " << cacheRate << "\n"
                  << "numDensifyTrials: " << numDensifyTrials << "\n";
    }
};

// Generate random document data given the centroids and std devs.
std::pair<std::vector<T_DOC_IDX>, std::vector<std::vector<T_EMB>>> genRandDocData(
    const ExpSetting& s,
    const std::vector<std::vector<float>>& centroidEmbs,
    const std::vector<std::vector<float>>& centroidStdDevs)
{
    size_t numCentroids = centroidEmbs.size();
    std::vector<T_DOC_IDX> docIdxList(s.numDocs);
    std::vector<std::vector<T_EMB>> emb2D(s.numDocs);
    int numThreads = omp_get_max_threads();
    std::vector<std::default_random_engine> generators(numThreads);
    for (int t = 0; t < numThreads; ++t)
    {
        generators.at(t).seed(t);
    }
#pragma omp parallel for schedule(static)
    for (size_t docIdx = 0; docIdx < s.numDocs; ++docIdx)
    {
        std::default_random_engine& generator = generators.at(omp_get_thread_num());
        docIdxList.at(docIdx) = docIdx;
        emb2D.at(docIdx).resize(s.totalEmbDim);
        int centroidIdx = docIdx % numCentroids;
        for (size_t embIdx = 0; embIdx < s.totalEmbDim; ++embIdx)
        {
            float centroid = centroidEmbs.at(centroidIdx).at(embIdx);
            float stdDev = centroidStdDevs.at(centroidIdx).at(embIdx);
            std::normal_distribution<float> distribution(centroid, stdDev);
            float randVal = distribution(generator);
            randVal = std::min(randVal, centroid + 3 * stdDev);
            randVal = std::max(randVal, centroid - 3 * stdDev);
            emb2D.at(docIdx).at(embIdx) = (T_EMB)randVal;
        }
    }
    return { docIdxList, emb2D };
}

// Generate the next docIdxList to densify based on the current docIdxList and the cache rate.
// If the cache rate is 0.9, then 90% of the current docIdxList will be kept, while the remaining 10% will be randomly
// generated.
std::pair<std::vector<T_DOC_IDX>, float> genNextDocIdxList(const ExpSetting& s,
                                                           const std::vector<T_DOC_IDX>& current,
                                                           int trial)
{
    static std::default_random_engine generator(trial);
    size_t numToKeep = static_cast<size_t>(s.numDocsToDensify * s.cacheRate);
    std::uniform_int_distribution<T_DOC_IDX> dist(0, s.numDocs - 1);

    std::unordered_set<T_DOC_IDX> nextSet(current.begin(), current.begin() + numToKeep);
    while (nextSet.size() < s.numDocsToDensify)
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

    float cacheRate = static_cast<float>(overlapCount) / s.numDocsToDensify;
    std::vector<T_DOC_IDX> next(nextSet.begin(), nextSet.end());
    std::sort(next.begin(), next.end());
    return { next, cacheRate };
}

// Generate a random docIdxList to densify.
std::vector<T_DOC_IDX> genRandomDocIdxList(const ExpSetting& s)
{
    std::unordered_set<T_DOC_IDX> docIdxSet;
    std::default_random_engine generator;
    std::uniform_int_distribution<T_DOC_IDX> distribution(0, s.numDocs - 1);
    while (docIdxSet.size() < s.numDocsToDensify)
    {
        docIdxSet.insert(distribution(generator));
    }
    std::vector<T_DOC_IDX> docIdxList(docIdxSet.begin(), docIdxSet.end());
    std::sort(docIdxList.begin(), docIdxList.end());
    return docIdxList;
}

// Generate random centroids and std devs.
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> genRandCentroids(const ExpSetting& s)
{
    std::default_random_engine generator(42);
    std::normal_distribution<float> distribution(s.centroidMean, s.centroidStdDev);
    std::vector<std::vector<float>> centroidEmbs(s.numCentroids, std::vector<float>(s.totalEmbDim));
    std::vector<std::vector<float>> centroidStdDevs(s.numCentroids,
                                                    std::vector<float>(s.totalEmbDim, s.centroidStdDev));
    for (size_t c = 0; c < s.numCentroids; c++)
    {
        for (size_t d = 0; d < s.totalEmbDim; d++)
        {
            centroidEmbs.at(c).at(d) = std::min(100.0f, std::max(-100.0f, distribution(generator)));
        }
    }
    return std::make_pair(centroidEmbs, centroidStdDevs);
}

bool isCompressedDim(const ExpSetting& s, size_t embIdx)
{
    for (const auto& config : s.residentPartitionConfigs)
    {
        if (embIdx >= config.getEmbDimBeginIncl() && embIdx < config.getEmbDimEndExcl())
        {
            return false;
        }
    }
    return true;
}

// Verify the densification result.
float verifyDensification(const ExpSetting& s,
                          const WorkingEmbDataset& workingEmbDataset,
                          const std::vector<T_DOC_IDX>& docIdxList,
                          const std::vector<std::vector<T_EMB>>& emb2D)
{
    size_t densifiedEmbDim = s.densifiedEmbIdxEndExcl - s.densifiedEmbIdxBeginIncl;
    std::vector<T_EMB> v_workingEmbDataset(s.maxWorkingSetSize * s.totalEmbDim);
    CHECK_CUDA(cudaMemcpy(v_workingEmbDataset.data(),
                          workingEmbDataset.data(),
                          s.numDocsToDensify * densifiedEmbDim * sizeof(T_EMB),
                          cudaMemcpyDeviceToHost));

    float compressedErrorSum = 0.0f;
    size_t compressedCount = 0;

    for (size_t docIdx = 0; docIdx < docIdxList.size(); ++docIdx)
    {
        for (size_t embIdx = s.densifiedEmbIdxBeginIncl; embIdx < s.densifiedEmbIdxEndExcl; ++embIdx)
        {
            size_t memAddr
                = getMemAddrRowMajor(docIdx, embIdx - s.densifiedEmbIdxBeginIncl, s.numDocsToDensify, densifiedEmbDim);
            float val = static_cast<float>(v_workingEmbDataset.at(memAddr));
            float ref = static_cast<float>(emb2D.at(docIdxList.at(docIdx)).at(embIdx));

            if (isCompressedDim(s, embIdx))
            {
                float error = std::abs(val - ref);
                compressedErrorSum += error;
                compressedCount++;
                if (error > 1.1f * s.centroidStdDev)
                {
                    std::ostringstream oss;
                    oss << "Compressed dim: docIdx = " << docIdx << ", embIdx = " << embIdx << ", val(" << val
                        << ") != ref(" << ref << ")"
                        << ", error = " << error << ", threshold = " << s.centroidStdDev;
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

void runExp(ExpSetting s)
{
    s.print();

    // ------------------
    // Prepare the data

    // --------
    // Generate the centroids
    auto [centroidEmbs, centroidStdDevs] = genRandCentroids(s);

    // --------
    // Generate 2D embeddings and centroid indices
    auto [docIdxList, emb2D] = genRandDocData(s, centroidEmbs, centroidStdDevs);

    // ------------------
    // Initialize the EmbDatasetManager
    EmbDatasetManager embDatasetManager(s.numDocs,
                                        s.totalEmbDim,
                                        s.residentPartitionConfigs,
                                        s.maxWorkingSetSize,
                                        s.numBitsPerDim,
                                        centroidEmbs,
                                        centroidStdDevs);
    // --------
    // Ingest the data into the EmbDatasetManager
    embDatasetManager.update(docIdxList, emb2D);

    // --------
    // Generate random docs to densify
    std::vector<T_DOC_IDX> docIdxListToDensify = genRandomDocIdxList(s);

    // --------
    // Some statistics
    float totalDensifyTimeMs = 0.0f;
    float totalCompressedError = 0.0f;
    float totalCacheRate = 0.0f;

    // --------
    // Densify the data
    for (int trial = -3; trial < s.numDensifyTrials; ++trial)
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
                                                                               s.densifiedEmbIdxBeginIncl,
                                                                               s.densifiedEmbIdxEndExcl,
                                                                               MemLayout::ROW_MAJOR);
        totalDensifyTimeMs += timer.tocMs();

        // --------
        // Verify the result
        totalCompressedError += verifyDensification(s, workingEmbDataset, docIdxListToDensify, emb2D);

        // --------
        // Generate the next docIdxList to densify
        auto [nextList, cacheRate] = genNextDocIdxList(s, docIdxListToDensify, trial);
        totalCacheRate += cacheRate;
        docIdxListToDensify = std::move(nextList);
    }

    // --------
    // Print the statistics
    std::cout << std::fixed << std::setprecision(3) << "\n===== Summary (avg over " << s.numDensifyTrials
              << " trials) =====\n"
              << "Densification time: " << totalDensifyTimeMs / s.numDensifyTrials << " ms\n"
              << "Cache rate: " << totalCacheRate / s.numDensifyTrials << "\n";
    std::cout << std::setprecision(6) << "Compressed dim avg error: " << totalCompressedError / s.numDensifyTrials
              << "\n";
    std::cout << "\n===== Time Record =====\n";
    embDatasetManager.getLastTimeRecordAndReset().print();
}

int main()
{
    if (!hasCudaDevice())
    {
        return 0;
    }

    runExp(ExpSetting {});

    return 0;
}
