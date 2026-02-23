#include <stack>
#include <unordered_set>

#include "common/const.hpp"
#include "manager/emb_dataset_manager.hpp"
#include "utils/util.hpp"

namespace // Anonymous namespace to avoid polluting the global namespace.
{
constexpr bool kDebug = false; // Set to true to perform some slow checks.
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
        if (reorderedDocIdxList.size() != desiredDocIdxList.size())
        {
            std::ostringstream oss;
            oss << "reorderedDocIdxList.size() != desiredDocIdxList.size(): " << reorderedDocIdxList.size() << " != " << desiredDocIdxList.size();
            throw std::runtime_error(oss.str());
        }
    }

    // ------------
    // Reassign the reorderedDocIdxList to the docIdxList.
    {
        Timer timer;
        timer.tic();
        desiredDocIdxList = std::move(reorderedDocIdxList);
        m_lastTimeRecord.cacheReassignMs += timer.tocMs();
    }

}
