#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stack>
#include <unordered_set>

#include "common/typedef.hpp"
#include "manager/emb_dataset_manager.hpp"
#include "utils/util.hpp"

namespace // Anonymous namespace to avoid polluting the global namespace.
{
constexpr T_DOC_IDX kInvalidDocIdx = -1; // Use -1 to indicate an invalid index in caching.
constexpr bool kDebug = false; // Set to true to perform some slow checks.
}

void TimeRecord::print() const
{
    float n = static_cast<float>(count);
    std::cout << std::fixed << std::setprecision(3) << "[densify] count: " << count << "\n"
              << "[densify] total: " << densifyTotalMs / n << " ms avg\n"
              << "[densify] total - cache: " << (densifyTotalMs - densifyCacheMs) / n << " ms avg\n"
              << "[densify] cache: " << densifyCacheMs / n << " ms avg\n"
              << "          [cache] first scan: " << cacheFirstScanMs / n << " ms avg\n"
              << "          [cache] second scan: " << cacheSecondScanMs / n << " ms avg\n"
              << "          [cache] reassign: " << cacheReassignMs / n << " ms avg\n"
              << "[densify] copy tasks: " << densifyCopyTasksMs / n << " ms avg\n"
              << "[densify] cudaMemcpy docIdxList H2D: " << densifyMemcpyH2DMs / n << " ms avg\n";
    for (size_t i = 0; i < densifyResidentPartitionMs.size(); ++i)
    {
        std::cout << "[densify] residentPartition[" << i << "]: " << densifyResidentPartitionMs[i] / n << " ms avg\n";
    }
    std::cout << "[densify] densifyCompressed: " << densifyCompressedMs / n << " ms avg\n";
}

TimeRecord EmbDatasetManager::getLastTimeRecordAndReset()
{
    TimeRecord record = m_lastTimeRecord;
    m_lastTimeRecord = TimeRecord {};
    return record;
}

EmbDatasetManager::EmbDatasetManager(size_t numDocs,
                                     size_t totalEmbDim,
                                     std::vector<ResidentPartitionConfig> residentPartitionConfigs,
                                     size_t maxNumWorkingDocs,
                                     size_t numBitsPerDim,
                                     const std::vector<std::vector<float>>& centroidEmbs,
                                     const std::vector<std::vector<float>>& centroidStdDevs)
    : m_numDocs(numDocs)
    , m_totalEmbDim(totalEmbDim)
    , m_maxNumWorkingDocs(maxNumWorkingDocs)
    , m_resQuantDataset(numDocs,
                        totalEmbDim,
                        maxNumWorkingDocs,
                        findCompressedPartitionConfigs(residentPartitionConfigs, totalEmbDim),
                        numBitsPerDim,
                        centroidEmbs,
                        centroidStdDevs)
    , m_workingEmbDataset(maxNumWorkingDocs, totalEmbDim)
    , m_d_docIdxListToDensify(maxNumWorkingDocs, "m_docIdxListToDensify")
    , m_h_copyTasks(maxNumWorkingDocs, "m_h_copyTasks")
    , m_d_copyTasks(maxNumWorkingDocs, "m_d_copyTasks")
    , m_currDocIdxListInWorkingDataset(maxNumWorkingDocs, kInvalidDocIdx)
{
    // --------------
    // We don't really need this sort, but we just do it for convenience.
    std::sort(residentPartitionConfigs.begin(), residentPartitionConfigs.end());

    // --------------
    // Confirm the resident partitions are disjoint in embDim space.
    for (size_t i = 1; i < residentPartitionConfigs.size(); ++i)
    {
        if (residentPartitionConfigs.at(i - 1).getEmbDimEndExcl() > residentPartitionConfigs.at(i).getEmbDimBeginIncl())
        {
            std::ostringstream oss;
            oss << "Resident partition configs have overlapping embedding dimension ranges: "
                << residentPartitionConfigs.at(i - 1).getEmbDimEndExcl() << " > "
                << residentPartitionConfigs.at(i).getEmbDimBeginIncl();
            throw std::runtime_error(oss.str());
        }
    }

    // --------------
    // Initialize the resident datasets.
    for (const auto& residentPartitionConfig : residentPartitionConfigs)
    {
        m_residentEmbDatasets.push_back(ResidentEmbDataset(numDocs, residentPartitionConfig));
    }
}

void EmbDatasetManager::update(const std::vector<T_DOC_IDX>& docIdxList,
                               const std::vector<std::vector<T_EMB>>& emb2D,
                               const std::vector<int>& centroidIdxList)
{
    // --------------
    // Some input sanity checks.
    if (docIdxList.size() > m_numDocs)
    {
        std::ostringstream oss;
        oss << "docIdxList.size() > m_numDocs: " << docIdxList.size() << " > " << m_numDocs;
        throw std::runtime_error(oss.str());
    }

    // --------------
    // Update the resident datasets.
    for (size_t residentPartitionIdx = 0; residentPartitionIdx < m_residentEmbDatasets.size(); ++residentPartitionIdx)
    {
        auto& embDataset = m_residentEmbDatasets[residentPartitionIdx];
        embDataset.update(docIdxList, emb2D);
    }

    // --------------
    // Update the compressed dataset.
    m_resQuantDataset.update(docIdxList, emb2D, centroidIdxList);
}

const WorkingEmbDataset& EmbDatasetManager::densify(std::vector<T_DOC_IDX>& docIdxList,
                                                    size_t embIdxBeginIncl,
                                                    size_t embIdxEndExcl,
                                                    MemLayout memLayout)
{
    // --------------
    // Input sanity checks.
    if (docIdxList.size() > m_maxNumWorkingDocs)
    {
        std::ostringstream oss;
        oss << "docIdxList.size() > m_maxNumWorkingDocs: " << docIdxList.size() << " > " << m_maxNumWorkingDocs;
        throw std::runtime_error(oss.str());
    }

    // --------------
    // Prepare timers
    Timer e2eTimer;
    {
        m_lastTimeRecord.count++;
        e2eTimer.tic();
    }

    // ------------
    // Cache the docIdxList.
    {
        Timer timer;
        timer.tic();
        cache(docIdxList);
        m_lastTimeRecord.densifyCacheMs += timer.tocMs();
    }

    // ------------
    // Copy the copy tasks to the device.
    {
        Timer timer;
        timer.tic();
        CHECK_CUDA(cudaMemcpy(m_d_copyTasks.data(), m_h_copyTasks.data(), m_numCopyTasks * sizeof(CopyTask), cudaMemcpyHostToDevice));
        m_lastTimeRecord.densifyCopyTasksMs += timer.tocMs();
    }

    // --------------
    // Copy the docIdxList to the device.
    {
        Timer timer;
        timer.tic();
        CHECK_CUDA(cudaMemcpy(m_d_docIdxListToDensify.data(),
                              docIdxList.data(),
                              docIdxList.size() * sizeof(T_DOC_IDX),
                              cudaMemcpyHostToDevice));
        m_lastTimeRecord.densifyMemcpyH2DMs += timer.tocMs();
    }

    // --------------
    // Set the working dataset properties.
    {
        m_workingEmbDataset.setMemLayout(memLayout);
        m_workingEmbDataset.setEmbDimBeginIncl(embIdxBeginIncl);
        m_workingEmbDataset.setEmbDimEndExcl(embIdxEndExcl);
    }

    // --------------
    // Prepare the densification task.
    DensificationTask densificationTask;
    {
        densificationTask.numDocsToDensify = docIdxList.size();
        densificationTask.numCopyTasks = m_numCopyTasks;
        densificationTask.globalEmbIdxBeginIncl = embIdxBeginIncl;
        densificationTask.globalEmbIdxEndExcl = embIdxEndExcl;
        densificationTask.d_workingEmbDataset = m_workingEmbDataset.data();
        densificationTask.d_docIdxList = m_d_docIdxListToDensify.data();
        densificationTask.d_copyTasks = m_d_copyTasks.data();
    }

    // --------------
    // Densify the resident datasets.
    {
        m_lastTimeRecord.densifyResidentPartitionMs.resize(m_residentEmbDatasets.size());
        for (size_t residentPartitionIdx = 0; residentPartitionIdx < m_residentEmbDatasets.size();
             ++residentPartitionIdx)
        {
            Timer timer;
            timer.tic();
            const auto& embDataset = m_residentEmbDatasets[residentPartitionIdx];
            embDataset.densify(densificationTask);
            m_lastTimeRecord.densifyResidentPartitionMs[residentPartitionIdx] += timer.tocMs();
        }
    }
    // --------------
    // Densify the compressed dataset.
    {
        Timer timer;
        timer.tic();
        m_resQuantDataset.densifyCompressed(densificationTask);
        m_lastTimeRecord.densifyCompressedMs += timer.tocMs();
    }

    // --------------
    // Record the end time.
    {
        m_lastTimeRecord.densifyTotalMs += e2eTimer.tocMs();
    }

    return m_workingEmbDataset;
}

// Please check the example in the end of this file to understand the caching logic.
void EmbDatasetManager::cache(std::vector<T_DOC_IDX>& desiredDocIdxList)
{
    // ------------
    // Verify desiredDocIdxList is unique (only run in debug mode)
    if (kDebug)
    {
        std::unordered_set<T_DOC_IDX> seen;
        for (T_DOC_IDX docIdx : desiredDocIdxList)
        {
            if (!seen.insert(docIdx).second)
            {
                std::ostringstream oss;
                oss << "Duplicate docIdx in desiredDocIdxList: " << docIdx;
                throw std::runtime_error(oss.str());
            }
        }
    }

    // ------------
    // First scan to find the cached working indices.
    std::vector<T_DOC_IDX> reorderedDocIdxList(desiredDocIdxList.size(), kInvalidDocIdx);
    std::stack<T_DOC_IDX> uncachedDocIndices;
    m_numCopyTasks = 0;

     // these two counters are for some sanity checks later. They will be ++ in step 1 and -- in step 2.
     // At the end both should be 0.
    int cntCached = 0;
    int cntUncached = 0;
    {
        Timer timer;
        timer.tic();
        for (T_DOC_IDX desiredWorkingIdx = 0; desiredWorkingIdx < desiredDocIdxList.size(); ++desiredWorkingIdx)
        {
            T_DOC_IDX desiredDocIdx = desiredDocIdxList.at(desiredWorkingIdx);
            auto it = m_currDocIdxToWorkingIdx.find(desiredDocIdx);
            if (it != m_currDocIdxToWorkingIdx.end() && it->second < desiredDocIdxList.size()) // "->second" is the "current working index"
            {
                T_DOC_IDX currWorkingIdx = it->second;
                reorderedDocIdxList.at(currWorkingIdx) = desiredDocIdx;
                cntCached++;
            }
            else
            {
                // Very important: if the cached working index is larger than the docIdxList.size(),
                //                 it is still considered as uncached.
                uncachedDocIndices.push(desiredDocIdx);
                cntUncached++;
            }
        }
        m_lastTimeRecord.cacheFirstScanMs += timer.tocMs();
    }

    // ------------
    // Verify no two docIdx map to the same cachedWorkingIdx. (only run in debug mode)
    if (kDebug)
    {
        std::unordered_set<T_DOC_IDX> seenWorkingIdx;
        for (T_DOC_IDX docIdx : desiredDocIdxList)
        {
            auto it = m_currDocIdxToWorkingIdx.find(docIdx);
            if (it != m_currDocIdxToWorkingIdx.end() && it->second < desiredDocIdxList.size())
            {
                if (!seenWorkingIdx.insert(it->second).second)
                {
                    std::ostringstream oss;
                    oss << "Two docIdx values map to same cachedWorkingIdx: " << it->second << " (docIdx=" << docIdx
                        << ")";
                    throw std::runtime_error(oss.str());
                }
            }
        }
    }

    // ------------
    // Second scan to put the uncached doc indices to the reorderedDocIdxList.
    {
        Timer timer;
        timer.tic();
        for (T_DOC_IDX workingIdx = 0; workingIdx < reorderedDocIdxList.size(); ++workingIdx)
        {
            if (reorderedDocIdxList.at(workingIdx) == kInvalidDocIdx) // It means this working index is an "empty slot"
            {
                // ---------
                // Find a uncached doc index and put it in the empty slot, and then pop it
                T_DOC_IDX uncachedDocIdx = uncachedDocIndices.top();
                reorderedDocIdxList.at(workingIdx) = uncachedDocIdx;
                uncachedDocIndices.pop();

                // ---------
                // Find which doc this working index is currently pointing to, and remove it from the m_currDocIdxToWorkingIdx map
                T_DOC_IDX oldDocIdx = m_currDocIdxListInWorkingDataset.at(workingIdx);
                if (oldDocIdx != kInvalidDocIdx)
                {
                    m_currDocIdxToWorkingIdx.erase(oldDocIdx);
                }

                // ---------
                // Update the m_currDocIdxToWorkingIdx to make uncachedDocIdx point to this working index
                // Also update the m_currDocIdxInWorkingDataset to make this working index point to uncachedDocIdx
                m_currDocIdxToWorkingIdx[uncachedDocIdx] = workingIdx;
                m_currDocIdxListInWorkingDataset.at(workingIdx) = uncachedDocIdx;
                m_h_copyTasks.data()[m_numCopyTasks++] = CopyTask { uncachedDocIdx, workingIdx };

                // ---------
                // Decrease the counter - meaning "we have dealt with one more uncached doc index"
                cntUncached--;
            }
            else // It means this working index is already occupied (i.e., cached)
            {
                // ---------
                // Decrease the counter - meaning "we have seen one more cached doc index"
                cntCached--;
            }
        }
        m_lastTimeRecord.cacheSecondScanMs += timer.tocMs();
    }

    // ------------
    // Verifications
    {
        if (!uncachedDocIndices.empty())
        {
            std::ostringstream oss;
            oss << "Uncached doc indices are not empty: " << uncachedDocIndices.size();
            throw std::runtime_error(oss.str());
        }
        if (cntCached != 0)
        {
            std::ostringstream oss;
            oss << "cntCached != 0: " << cntCached;
            throw std::runtime_error(oss.str());
        }
        if (cntUncached != 0)
        {
            std::ostringstream oss;
            oss << "cntUncached != 0: " << cntUncached;
            throw std::runtime_error(oss.str());
        }
    }

    // ------------
    // Reassign the reorderedDocIdxList to the docIdxList.
    {
        Timer timer;
        timer.tic();
        desiredDocIdxList = reorderedDocIdxList;
        m_lastTimeRecord.cacheReassignMs += timer.tocMs();
    }

}

/*
Let's say we have the following densification task:
  desiredDocIdxList = [2, 6, 9, 13]

And in the current WorkingEmbDataset, we have the following doc indices (here we assume maxNumWorkingDocs = 5)
  currDocIdxList = [2, 5, 6, 10, 13]

Naively thinking, you may think all 2, 6, 13 are already cached (YAY!). Unfortunately, it is not the case.
In this example, only 2 is cached. 6 and 13 are not cached because they are at different positions in the
WorkingEmbDataset. (In desiredDocIdxList, 6 is at position 1, while in currDocIdxList, 6 is at position 2.)

That is to say, the precise definition of "cached" is NOT just "the desired doc is in the WorkingEmbDataset". It's more
strict than that - it has to be "the desired doc is in the WorkingEmbDataset AND at the desired position"

The naive solution is to just copy the embeddings of doc 6 from position 2 to position 1 in the WorkingEmbDataset.
However, such copy will take time. Is there a way to avoid copying doc 6 at all?

Yes, the answer is we alter the desiredDocIdxList.
So basically, in this example we want to re-order the desiredDocIdxList to [2, 9, 6, 13]. This way, we have both doc 2
and 6 cached.

It is important to note that doc 13 will never be cached. The reason is that in currDocIdxList, doc 13 is at position
4, but the length of desiredDocIdxList is 4. So there is no way we can re-order the desiredDocIdxList so that doc 13 is
at the position 4.

In order to achieve such re-ordering, we will do it in two steps:

==== Step 1 ====
We create a new reorderedDocIdxList with -1 as initial value and has the same length as desiredDocIdxList. Like this:

  reorderedDocIdxList = [-1, -1, -1, -1]

The we loop through desiredDocIdxList, and for each docIdx, we know
  1) if it is in currDocIdxList, and if yes,
  2) where it is in currDocIdxList,

So it will be like this (2 / 6 are found in currDocIdxList, while 9 and 13 are not):

  reorderedDocIdxList = [2, -1, 6, -1]

At the same time, we will maintain a list to record the uncached doc indices, which we will deal with in step 2.

  uncachedDocIdxList = [9, 13]

==== Step 2 ====
In this step, we want to put those uncached doc indices to the reorderedDocIdxList. This time we loop through the
reorderedDocIdxList, and every time we see a -1, it means it is an "empty slot", so then we pop the top of the
uncachedDocIdxList and put it there. Depending on you like to use stack or queue for the uncachedDocIdxList, you may get
either:

  reorderedDocIdxList = [2, 9, 6, 13] (using queue)
  reorderedDocIdxList = [2, 13, 9, 6] (using stack)

Both are okay.

Note that in the step, we will also record a list of "copy tasks" to indicate what should be copied from where to where
(for those uncached doc indices) like this:

  copyTasks = [(srcDocIdx=13, dstWokringIdx=1), (srcDocIdx=9, dstWokringIdx=3)]
  (assuming reorderedDocIdxList is [2, 13, 6, 9])

==== Final step ====
Finally, we reassign the reorderedDocIdxList to the desiredDocIdxList. In the densification kernels, we will only copy
the embeddings according to the copyTasks list. (So as you can see in this example, only 2 out of 4 embeddings (13, 9)
are copied.)
*/