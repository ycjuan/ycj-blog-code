#include "resident/resident_emb_dataset.hpp"
#include "utils/cuda_malloc_raii.hpp"
#include "utils/util.hpp"

ResidentEmbDataset::ResidentEmbDataset(size_t numDocs, ResidentPartitionConfig residentPartitionConfig)
    : m_numDocs(numDocs)
    , m_residentPartitionConfig(residentPartitionConfig)
    , m_residentEmbDataset(numDocs * residentPartitionConfig.getEmbDim(), "m_residentEmbDataset")
    , m_docIdxChunk(kMaxUpdateBatchSize, "m_docIdxChunk")
    , m_embChunk(kMaxUpdateBatchSize * residentPartitionConfig.getEmbDim(), "m_embChunk")
{
}

T_EMB* ResidentEmbDataset::data() const { return m_residentEmbDataset.data(); }

ResidentPartitionConfig ResidentEmbDataset::getResidentPartitionConfig() const { return m_residentPartitionConfig; }

struct DensifyFromResidentKernelParams
{
    // Source
    const T_EMB* d_residentEmbDataset;
    T_DOC_IDX* docIdxList;
    T_DOC_IDX numDocsTotal;
    ResidentPartitionConfig residentPartitionConfig;
    size_t embOffsetSrc;

    // Destination
    T_EMB* d_workingEmbDataset;
    int numDocsToDensify;
    size_t embOffsetDst;
    int embDimWorking;

    // Shared
    int embDimToDensify;
    T_DOC_IDX* d_docIdxMap;
    int8_t* hp_isCached;
};

__global__ void densifyFromResidentKernel(DensifyFromResidentKernelParams params)
{
    size_t taskIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t docIdxDst = taskIdx / params.embDimToDensify;
    size_t embIdx = taskIdx % params.embDimToDensify;

    if (docIdxDst < params.numDocsToDensify)
    {
        if (params.hp_isCached[docIdxDst])
        {
            return;
        }

        T_DOC_IDX docIdxSrc = params.d_docIdxMap[docIdxDst];

        size_t memAddrSrc = getMemAddrRowMajor(docIdxSrc,
                                               embIdx + params.embOffsetSrc,
                                               params.numDocsTotal,
                                               params.residentPartitionConfig.getEmbDim());
        size_t memAddrDst = getMemAddrRowMajor(docIdxDst,
                                               embIdx + params.embOffsetDst,
                                               params.numDocsToDensify,
                                               params.embDimWorking);
        params.d_workingEmbDataset[memAddrDst] = params.d_residentEmbDataset[memAddrSrc];
    }
}

void ResidentEmbDataset::densify(const DensificationTask& densificationTask) const
{
    // -------------
    // Obtain the real begin and end points we want to densify.
    const size_t embDimBeginInclReal = std::max(densificationTask.globalEmbIdxBeginIncl, m_residentPartitionConfig.getEmbDimBeginIncl());
    const size_t embDimEndExclReal = std::min(densificationTask.globalEmbIdxEndExcl, m_residentPartitionConfig.getEmbDimEndExcl());

    // -------------
    // Prepare the parameters for the kernel.
    DensifyFromResidentKernelParams params;
    params.d_residentEmbDataset = m_residentEmbDataset.data();
    params.d_docIdxMap = densificationTask.d_docIdxList;
    params.d_workingEmbDataset = densificationTask.d_workingEmbDataset;
    params.numDocsTotal = m_numDocs;
    params.residentPartitionConfig = m_residentPartitionConfig;
    params.numDocsToDensify = densificationTask.numDocsToDensify;
    params.embDimWorking = densificationTask.globalEmbIdxEndExcl - densificationTask.globalEmbIdxBeginIncl;
    params.embDimToDensify = embDimEndExclReal - embDimBeginInclReal;
    params.embOffsetSrc = embDimBeginInclReal - m_residentPartitionConfig.getEmbDimBeginIncl();
    params.embOffsetDst = embDimBeginInclReal - densificationTask.globalEmbIdxBeginIncl;
    params.hp_isCached = densificationTask.hp_isCached;
    // -------------
    // Launch the kernel.
    constexpr size_t kBlockSize = 1024;
    size_t numTasks = params.numDocsToDensify * params.embDimToDensify;
    size_t gridSize = (numTasks + kBlockSize - 1) / kBlockSize;
    densifyFromResidentKernel<<<gridSize, kBlockSize, 0, m_cudaStreamRead.get()>>>(params);
    CHECK_CUDA(cudaStreamSynchronize(m_cudaStreamRead.get()));
}

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
        T_DOC_IDX* h_docIdxChunk = m_docIdxChunk.data();
        T_EMB* h_embChunk = m_embChunk.data();
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
                                                                            m_residentEmbDataset.data(),
                                                                            m_numDocs);
        CHECK_CUDA(cudaStreamSynchronize(m_cudaStreamRead.get()));
    }
}