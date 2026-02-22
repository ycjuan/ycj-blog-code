#include "resident/resident_emb_dataset.hpp"
#include "utils/cuda_raii.hpp"
#include "utils/util.hpp"

ResidentEmbDataset::ResidentEmbDataset(size_t numDocs, ResidentPartitionConfig residentPartitionConfig)
    : m_numDocs(numDocs)
    , m_residentPartitionConfig(residentPartitionConfig)
    , m_d_embData(numDocs * residentPartitionConfig.getEmbDim(), "m_residentEmbDataset")
    , m_h_docIdxChunk(kMaxUpdateBatchSize, "m_docIdxChunk")
    , m_h_embDataChunk(kMaxUpdateBatchSize * residentPartitionConfig.getEmbDim(), "m_embChunk")
{
}

T_EMB* ResidentEmbDataset::data() const { return m_d_embData.data(); }

ResidentPartitionConfig ResidentEmbDataset::getResidentPartitionConfig() const { return m_residentPartitionConfig; }

__global__ void updateResidentKernel(T_DOC_IDX* h_docIdxChunk,
                                     T_EMB* h_embChunk,
                                     size_t numDocsToUpdate,
                                     size_t embDim,
                                     T_EMB* d_residentEmbDataset,
                                     size_t numDocsTotal)
{
    size_t taskIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (taskIdx < numDocsToUpdate)
    {
        size_t srcDocIdx = taskIdx;
        T_DOC_IDX dstDocIdx = h_docIdxChunk[srcDocIdx];
        for (size_t embIdx = 0; embIdx < embDim; ++embIdx)
        {
            size_t srcMemAddr = getMemAddrRowMajor(srcDocIdx, embIdx, numDocsToUpdate, embDim);
            size_t dstMemAddr = getMemAddrRowMajor(dstDocIdx, embIdx, numDocsTotal, embDim);
            d_residentEmbDataset[dstMemAddr] = h_embChunk[srcMemAddr];
        }
    }
}

void ResidentEmbDataset::update(const std::vector<T_DOC_IDX>& docIdxList, const std::vector<std::vector<T_EMB>>& emb2D)
{
    // Loop over docs chunk by chunk.
    for (size_t docIdxBeginIncl = 0; docIdxBeginIncl < docIdxList.size(); docIdxBeginIncl += kMaxUpdateBatchSize)
    {
        // -----------
        // Handle corner case to get the end index, and calculate real number of docs to update.
        size_t docIdxEndExcl = std::min(docIdxBeginIncl + kMaxUpdateBatchSize, docIdxList.size());
        size_t numDocsToUpdate = docIdxEndExcl - docIdxBeginIncl;

        // -------------
        // Copy data to the buffers
        T_DOC_IDX* h_docIdxChunk = m_h_docIdxChunk.data();
        T_EMB* h_embChunk = m_h_embDataChunk.data();
        for (size_t docIdx = 0; docIdx < numDocsToUpdate; ++docIdx)
        {
            h_docIdxChunk[docIdx] = docIdxList.at(docIdx + docIdxBeginIncl);

            for (size_t embIdx = 0; embIdx < m_residentPartitionConfig.getEmbDim(); ++embIdx)
            {
                size_t dstMemAddr
                    = getMemAddrRowMajor(docIdx, embIdx, kMaxUpdateBatchSize, m_residentPartitionConfig.getEmbDim());
                h_embChunk[dstMemAddr]
                    = emb2D.at(docIdx + docIdxBeginIncl).at(embIdx + m_residentPartitionConfig.getEmbDimBeginIncl());
            }
        }

        static constexpr size_t kBlockSize = 1024;
        size_t gridSize = (numDocsToUpdate + kBlockSize - 1) / kBlockSize;

        updateResidentKernel<<<gridSize, kBlockSize, 0, m_cudaStreamRead.get()>>>(h_docIdxChunk,
                                                                            h_embChunk,
                                                                            numDocsToUpdate,
                                                                            m_residentPartitionConfig.getEmbDim(),
                                                                            m_d_embData.data(),
                                                                            m_numDocs);
        CHECK_CUDA(cudaStreamSynchronize(m_cudaStreamRead.get()));
    }
}