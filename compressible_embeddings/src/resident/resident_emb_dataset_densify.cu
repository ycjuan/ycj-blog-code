#include "resident/resident_emb_dataset.hpp"
#include "utils/util.hpp"

struct DensifyFromResidentKernelParams
{
    // Source
    const T_EMB* d_srcEmbData;
    T_DOC_IDX srcNumDocs;
    ResidentPartitionConfig residentPartitionConfig;
    int srcEmbOffset;

    // Destination
    T_EMB* d_dstEmbData;
    int dstNumDocs;
    int dstEmbOffset;
    int dstEmbDim;

    // Shared
    int embDimToDensify;
    CopyTask* d_copyTasks;
    int numCopyTasks;
};

__global__ void densifyFromResidentKernel(DensifyFromResidentKernelParams params)
{
    size_t taskIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t copyTaskIdx = taskIdx / params.embDimToDensify;
    size_t embIdx = taskIdx % params.embDimToDensify;

    if (copyTaskIdx < params.numCopyTasks)
    {
        CopyTask task = params.d_copyTasks[copyTaskIdx];

        size_t memAddrSrc = getMemAddrRowMajor(task.srcDocIdx,
                                               embIdx + params.srcEmbOffset,
                                               params.srcNumDocs,
                                               params.residentPartitionConfig.getEmbDim());
        size_t memAddrDst = getMemAddrRowMajor(task.dstDocIdx,
                                               embIdx + params.dstEmbOffset,
                                               params.dstNumDocs,
                                               params.dstEmbDim);
        params.d_dstEmbData[memAddrDst] = params.d_srcEmbData[memAddrSrc];
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
    params.d_srcEmbData = m_d_embData.data();
    params.d_dstEmbData = densificationTask.d_workingEmbDataset;
    params.srcNumDocs = m_numDocs;
    params.residentPartitionConfig = m_residentPartitionConfig;
    params.dstNumDocs = densificationTask.numDocsToDensify;
    params.dstEmbDim = densificationTask.globalEmbIdxEndExcl - densificationTask.globalEmbIdxBeginIncl;
    params.embDimToDensify = embDimEndExclReal - embDimBeginInclReal;
    params.srcEmbOffset = embDimBeginInclReal - m_residentPartitionConfig.getEmbDimBeginIncl();
    params.dstEmbOffset = embDimBeginInclReal - densificationTask.globalEmbIdxBeginIncl;
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
