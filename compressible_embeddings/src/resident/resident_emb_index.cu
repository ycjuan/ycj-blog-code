#include "resident/resident_emb_index.hpp"
#include "utils/cuda_malloc_raii.hpp"
#include "utils/util.hpp"

ResidentEmbIndex::ResidentEmbIndex(size_t numDocs, ResidentPartitionConfig residentPartitionConfig)
    : m_numDocs(numDocs)
    , m_residentPartitionConfig(residentPartitionConfig)
    , m_residentEmbIndex(numDocs * residentPartitionConfig.getEmbDim(), "m_residentEmbIndex")
    , m_docIdxChunk(kMaxUpdateBatchSize, "m_docIdxChunk")
    , m_embChunk(kMaxUpdateBatchSize * residentPartitionConfig.getEmbDim(), "m_embChunk")
{
}

T_EMB* ResidentEmbIndex::data() const { return m_residentEmbIndex.data(); }

ResidentPartitionConfig ResidentEmbIndex::getResidentPartitionConfig() const { return m_residentPartitionConfig; }

struct DensifyFromResidentKernelParams
{
    // Source
    const T_EMB* d_residentEmbIndex;
    T_DOC_IDX* docIdxList;
    T_DOC_IDX numDocsTotal;
    ResidentPartitionConfig residentPartitionConfig;
    size_t embOffsetSrc;

    // Destination
    T_EMB* d_workingEmbIndex;
    int numDocsToDensify;
    size_t embOffsetDst;
    int embDimWorking;

    // Shared
    int embDimToDensify;
    T_DOC_IDX* d_docIdxMap;
};

__global__ void densifyFromResidentKernel(DensifyFromResidentKernelParams params)
{
    size_t taskIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (taskIdx < params.numDocsToDensify)
    {
        T_DOC_IDX docIdxSrc = params.d_docIdxMap[taskIdx];
        T_DOC_IDX docIdxDst = taskIdx;

        for (size_t embIdx = 0; embIdx < params.embDimToDensify; embIdx++)
        {
            size_t memAddrSrc = getMemAddrRowMajor(docIdxSrc,
                                                   embIdx + params.embOffsetSrc,
                                                   params.numDocsTotal,
                                                   params.residentPartitionConfig.getEmbDim());
            size_t memAddrDst = getMemAddrRowMajor(docIdxDst,
                                                   embIdx + params.embOffsetDst,
                                                   params.numDocsToDensify,
                                                   params.embDimWorking);
            params.d_workingEmbIndex[memAddrDst] = params.d_residentEmbIndex[memAddrSrc];
            if (abs((static_cast<float>(params.d_residentEmbIndex[memAddrSrc]) - 0.933594f)) < 1e-3f)
            {
                printf("docIdxSrc: %d, embIdx: %d, val: %f, docIdxDst: %d, embIdxDst: %d, numDocsToCopy: %d, embDimWorkingSetTotal: %d, dstMemAddr: %d\n",
                       static_cast<int>(docIdxSrc),
                       static_cast<int>(embIdx + params.embOffsetSrc),
                       static_cast<float>(params.d_residentEmbIndex[memAddrSrc]),
                       static_cast<int>(docIdxDst),
                       static_cast<int>(embIdx + params.embOffsetDst),
                       static_cast<int>(params.numDocsToDensify),
                       static_cast<int>(params.embDimWorking),
                       static_cast<int>(memAddrDst));
            }
        }
    }
}

void ResidentEmbIndex::densify(const DensificationTask& densificationTask) const
{
    // -------------
    // Obtain the real begin and end points we want to densify.
    const size_t embDimBeginInclReal = std::max(densificationTask.globalEmbIdxBeginIncl, m_residentPartitionConfig.getEmbDimBeginIncl());
    const size_t embDimEndExclReal = std::min(densificationTask.globalEmbIdxEndExcl, m_residentPartitionConfig.getEmbDimEndExcl());

    // -------------
    // Prepare the parameters for the kernel.
    DensifyFromResidentKernelParams params;
    params.d_residentEmbIndex = m_residentEmbIndex.data();
    params.d_docIdxMap = densificationTask.d_docIdxList;
    params.d_workingEmbIndex = densificationTask.d_workingEmbIndex;
    params.numDocsTotal = m_numDocs;
    params.residentPartitionConfig = m_residentPartitionConfig;
    params.numDocsToDensify = densificationTask.numDocsToDensify;
    params.embDimWorking = densificationTask.globalEmbIdxEndExcl - densificationTask.globalEmbIdxBeginIncl;
    params.embDimToDensify = embDimEndExclReal - embDimBeginInclReal;
    params.embOffsetSrc = embDimBeginInclReal - m_residentPartitionConfig.getEmbDimBeginIncl();
    params.embOffsetDst = embDimBeginInclReal - densificationTask.globalEmbIdxBeginIncl;
    std::cout << "embOffsetDst: " << params.embOffsetDst << std::endl;

    // -------------
    // Launch the kernel.
    std::cout << "densifyFromResidentKernel start" << std::endl;
    constexpr size_t kBlockSize = 1024;
    size_t gridSize = (params.numDocsToDensify + kBlockSize - 1) / kBlockSize;
    densifyFromResidentKernel<<<gridSize, kBlockSize, 0, m_cudaStreamRead.get()>>>(params);
    std::cout << "densifyFromResidentKernel done" << std::endl;
    CHECK_CUDA(cudaStreamSynchronize(m_cudaStreamRead.get()));
    std::cout << "cudaStreamSynchronize done" << std::endl;
}

__global__ void updateResidentKernel(T_DOC_IDX* h_docIdxChunk,
                                     T_EMB* h_embChunk,
                                     size_t numDocsToUpdate,
                                     size_t embDim,
                                     T_EMB* d_residentEmbIndex,
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
            d_residentEmbIndex[dstMemAddr] = h_embChunk[srcMemAddr];
        }
    }
}

void ResidentEmbIndex::update(const std::vector<T_DOC_IDX>& docIdxList, const std::vector<std::vector<T_EMB>>& emb2D)
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

        std::cout << "updateResidentKernel start" << std::endl;
        updateResidentKernel<<<gridSize, kBlockSize, 0, m_cudaStreamRead.get()>>>(h_docIdxChunk,
                                                                            h_embChunk,
                                                                            numDocsToUpdate,
                                                                            m_residentPartitionConfig.getEmbDim(),
                                                                            m_residentEmbIndex.data(),
                                                                            m_numDocs);
        std::cout << "updateResidentKernel done" << std::endl;
        CHECK_CUDA(cudaStreamSynchronize(m_cudaStreamRead.get()));
        std::cout << "cudaStreamSynchronize done" << std::endl;
    }
}