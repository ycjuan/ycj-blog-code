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
    T_EMB* d_workingSetEmbIndex;
    int numDocsToCopy;
    size_t embOffsetDst;
    int embDimWorkingSetTotal;

    // Shared
    int embDimToCopy;
    T_DOC_IDX* d_docIdxMap;
};

__global__ void densifyFromResidentKernel(DensifyFromResidentKernelParams params)
{
    size_t taskIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (taskIdx < params.numDocsToCopy)
    {
        T_DOC_IDX docIdxSrc = params.d_docIdxMap[taskIdx];
        T_DOC_IDX docIdxDst = taskIdx;

        for (size_t embIdx = 0; embIdx < params.embDimToCopy; embIdx++)
        {
            size_t memAddrSrc = getMemAddrRowMajor(docIdxSrc,
                                                   embIdx + params.embOffsetSrc,
                                                   params.numDocsTotal,
                                                   params.residentPartitionConfig.getEmbDim());
            size_t memAddrDst = getMemAddrRowMajor(docIdxDst,
                                                   embIdx + params.embOffsetDst,
                                                   params.numDocsToCopy,
                                                   params.embDimWorkingSetTotal);
            params.d_workingSetEmbIndex[memAddrDst] = params.d_residentEmbIndex[memAddrSrc];
            if (abs((static_cast<float>(params.d_residentEmbIndex[memAddrSrc]) - 0.933594f)) < 1e-3f)
            {
                printf("docIdxSrc: %d, embIdx: %d, val: %f, docIdxDst: %d, embIdxDst: %d, numDocsToCopy: %d, embDimWorkingSetTotal: %d, dstMemAddr: %d\n",
                       static_cast<int>(docIdxSrc),
                       static_cast<int>(embIdx + params.embOffsetSrc),
                       static_cast<float>(params.d_residentEmbIndex[memAddrSrc]),
                       static_cast<int>(docIdxDst),
                       static_cast<int>(embIdx + params.embOffsetDst),
                       static_cast<int>(params.numDocsToCopy),
                       static_cast<int>(params.embDimWorkingSetTotal),
                       static_cast<int>(memAddrDst));
            }
        }
    }
}

void ResidentEmbIndex::densify(const DensificationTask& densificationTask) const
{
    const size_t embDimBeginInclReal = std::max(densificationTask.embIdxBeginIncl, m_residentPartitionConfig.getEmbDimBeginIncl());
    const size_t embDimEndExclReal = std::min(densificationTask.embIdxEndExcl, m_residentPartitionConfig.getEmbDimEndExcl());

    DensifyFromResidentKernelParams params;
    params.d_residentEmbIndex = m_residentEmbIndex.data();
    params.d_docIdxMap = densificationTask.d_docIdxMap;
    params.d_workingSetEmbIndex = densificationTask.d_workingSetEmbIndex;
    params.numDocsTotal = m_numDocs;
    params.residentPartitionConfig = m_residentPartitionConfig;
    params.numDocsToCopy = densificationTask.numTasks;
    params.embDimWorkingSetTotal = densificationTask.embIdxEndExcl - densificationTask.embIdxBeginIncl;
    params.embDimToCopy = embDimEndExclReal - embDimBeginInclReal;
    params.embOffsetSrc = embDimBeginInclReal - m_residentPartitionConfig.getEmbDimBeginIncl();
    params.embOffsetDst = embDimBeginInclReal - densificationTask.embIdxBeginIncl;
    std::cout << "embOffsetDst: " << params.embOffsetDst << std::endl;

    std::cout << "densifyFromResidentKernel start" << std::endl;
    constexpr size_t kBlockSize = 1024;
    size_t gridSize = (params.numDocsToCopy + kBlockSize - 1) / kBlockSize;
    densifyFromResidentKernel<<<gridSize, kBlockSize, 0, m_cudaStreamRead>>>(params);
    std::cout << "densifyFromResidentKernel done" << std::endl;
    CHECK_CUDA(cudaStreamSynchronize(m_cudaStreamRead));
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
            d_residentEmbIndex[dstMemAddr] = h_embChunk[dstMemAddr];
        }
    }
}

void ResidentEmbIndex::update(const std::vector<T_DOC_IDX>& docIdxList, const std::vector<std::vector<T_EMB>>& emb2D)
{
    for (size_t docIdxBeginIncl = 0; docIdxBeginIncl < docIdxList.size(); docIdxBeginIncl += kMaxUpdateBatchSize)
    {
        size_t docIdxEndExcl = std::min(docIdxBeginIncl + kMaxUpdateBatchSize, docIdxList.size());
        size_t numDocsToUpdate = docIdxEndExcl - docIdxBeginIncl;
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
        updateResidentKernel<<<gridSize, kBlockSize, 0, m_cudaStreamRead>>>(h_docIdxChunk,
                                                                            h_embChunk,
                                                                            numDocsToUpdate,
                                                                            m_residentPartitionConfig.getEmbDim(),
                                                                            m_residentEmbIndex.data(),
                                                                            m_numDocs);
        std::cout << "updateResidentKernel done" << std::endl;
        CHECK_CUDA(cudaStreamSynchronize(m_cudaStreamRead));
        std::cout << "cudaStreamSynchronize done" << std::endl;
    }
}