#include "resident/resident_emb_dataset.hpp"
#include "utils/util.hpp"

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
    CopyTask* d_copyTasks;
    int numCopyTasks;
};

__global__ void densifyFromResidentKernel(DensifyFromResidentKernelParams params)
{
    size_t taskIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t copyIdx = taskIdx / params.embDimToDensify;
    size_t embIdx = taskIdx % params.embDimToDensify;

    if (copyIdx < params.numCopyTasks)
    {
        CopyTask task = params.d_copyTasks[copyIdx];
        T_DOC_IDX docIdxSrc = task.srcDocIdx;
        T_DOC_IDX docIdxDst = task.dstDocIdx;

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
    params.d_residentEmbDataset = m_d_embData.data();
    params.d_workingEmbDataset = densificationTask.d_workingEmbDataset;
    params.numDocsTotal = m_numDocs;
    params.residentPartitionConfig = m_residentPartitionConfig;
    params.numDocsToDensify = densificationTask.numDocsToDensify;
    params.embDimWorking = densificationTask.globalEmbIdxEndExcl - densificationTask.globalEmbIdxBeginIncl;
    params.embDimToDensify = embDimEndExclReal - embDimBeginInclReal;
    params.embOffsetSrc = embDimBeginInclReal - m_residentPartitionConfig.getEmbDimBeginIncl();
    params.embOffsetDst = embDimBeginInclReal - densificationTask.globalEmbIdxBeginIncl;
    params.d_copyTasks = densificationTask.d_copyTasks;
    params.numCopyTasks = densificationTask.numCopyTasks;
    // -------------
    // Launch the kernel.
    if (params.numCopyTasks == 0)
    {
        return;
    }
    constexpr size_t kBlockSize = 1024;
    size_t numTasks = params.numCopyTasks * params.embDimToDensify;
    size_t gridSize = (numTasks + kBlockSize - 1) / kBlockSize;
    densifyFromResidentKernel<<<gridSize, kBlockSize, 0, m_cudaStreamRead.get()>>>(params);
    CHECK_CUDA(cudaStreamSynchronize(m_cudaStreamRead.get()));
}
