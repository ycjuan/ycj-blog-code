#include <cassert>
#include <random>
#include <iostream>
#include <unordered_set>
#include <algorithm>

#include "manager/emb_index_manager.hpp"
#include "resident/resident_partition_config.hpp"
#include "working/working_emb_index.hpp"
#include "utils/util.hpp"

constexpr size_t kNumDocs = 10000;
constexpr size_t kTotalEmbDim = 128;
constexpr size_t kMaxWorkingSetSize = 1000;
constexpr size_t kNumDocsToDensify = 2;
constexpr size_t kDensifiedEmbIdxBeginIncl = 3;
constexpr size_t kDensifiedEmbIdxEndExcl = 125;
const std::vector<ResidentPartitionConfig> kResidentPartitionConfigs
    = { ResidentPartitionConfig(0, 48, MemLayout::ROW_MAJOR), ResidentPartitionConfig(64, 96, MemLayout::ROW_MAJOR) };

std::pair<std::vector<T_DOC_IDX>, std::vector<std::vector<T_EMB>>> populateRandomEmbIndex()
{
    std::vector<T_DOC_IDX> docIdxList(kNumDocs);
    std::vector<std::vector<T_EMB>> emb2D(kNumDocs);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for (size_t docIdx = 0; docIdx < kNumDocs; ++docIdx)
    {
        docIdxList.at(docIdx) = docIdx;
        emb2D.at(docIdx).resize(kTotalEmbDim);
        for (size_t embIdx = 0; embIdx < kTotalEmbDim; ++embIdx)
        {
            emb2D.at(docIdx).at(embIdx) = (T_EMB)distribution(generator);
        }
    }
    return std::make_pair(docIdxList, emb2D);
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

void verifyDensification(const WorkingEmbIndex& workingEmbIndex, const std::vector<T_DOC_IDX>& docIdxList, const std::vector<std::vector<T_EMB>>& emb2D)
{
    std::vector<T_EMB> v_workingEmbIndex(kMaxWorkingSetSize * kTotalEmbDim);
    CHECK_CUDA(cudaMemcpy(v_workingEmbIndex.data(),
                          workingEmbIndex.data(),
                          kNumDocsToDensify * (kDensifiedEmbIdxEndExcl - kDensifiedEmbIdxBeginIncl) * sizeof(T_EMB),
                          cudaMemcpyDeviceToHost));

    for (size_t docIdx = 0; docIdx < docIdxList.size(); ++docIdx)
    {
        for (size_t embIdx = kDensifiedEmbIdxBeginIncl; embIdx < kDensifiedEmbIdxEndExcl; ++embIdx)
        {
            if ( (embIdx >= 48 && embIdx < 64) || embIdx >= 96)
            {
                continue;
            }
            size_t memAddr = getMemAddrRowMajor(docIdx, embIdx - kDensifiedEmbIdxBeginIncl, kNumDocsToDensify, kDensifiedEmbIdxEndExcl - kDensifiedEmbIdxBeginIncl);
            auto val = v_workingEmbIndex.at(memAddr);
            auto ref = emb2D.at(docIdxList.at(docIdx)).at(embIdx);
            if (val != ref)
            {
                std::ostringstream oss;
                oss << "docIdx = " << docIdx << ", embIdx = " << embIdx << ", val(" << static_cast<float>(val)
                    << ") != ref(" << static_cast<float>(ref) << ")"
                    << ", memAddr: " << memAddr;
                throw std::runtime_error(oss.str());
            }
        }
    }

    printf("!!!!!!!!!!!! Densification verified successfully !!!!!!!!!!!!\n");
}

int main()
{
    EmbIndexManager embIndexManager(kNumDocs, kTotalEmbDim, kResidentPartitionConfigs, kMaxWorkingSetSize);

    auto [docIdxList, emb2D] = populateRandomEmbIndex();
    embIndexManager.update(docIdxList, emb2D);

    std::vector<T_DOC_IDX> docIdxListToDensify = genRandomDocIdxList();
    const WorkingEmbIndex& workingEmbIndex = embIndexManager.densify(docIdxListToDensify,
                                                                     kDensifiedEmbIdxBeginIncl,
                                                                     kDensifiedEmbIdxEndExcl,
                                                                     MemLayout::ROW_MAJOR);

    verifyDensification(workingEmbIndex, docIdxListToDensify, emb2D);

    return 0;
}