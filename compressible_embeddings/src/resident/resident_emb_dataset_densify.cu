#include "resident/resident_emb_dataset.hpp"
#include "utils/util.hpp"

struct DensifyFromResidentKernelParams
{
    // Source
    const T_EMB* d_srcEmbData;
    T_DOC_IDX srcNumDocs;
    int srcEmbDim;
    int srcEmbOffset;

    // Destination
    T_EMB* d_dstEmbData;
    int dstNumDocs;
    int dstEmbDim;
    int dstEmbOffset;

    // Copy tasks
    int embDimToCopy;
    CopyTask* d_copyTasks;
    int numCopyTasks;
};

__global__ void densifyFromResidentKernel(DensifyFromResidentKernelParams params)
{
    size_t taskIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t copyTaskIdx = taskIdx / params.embDimToCopy;
    size_t embIdx = taskIdx % params.embDimToCopy;

    if (copyTaskIdx < params.numCopyTasks)
    {
        CopyTask task = params.d_copyTasks[copyTaskIdx];

        size_t memAddrSrc = getMemAddrRowMajor(task.srcDocIdx,
                                               embIdx + params.srcEmbOffset,
                                               params.srcNumDocs,
                                               params.srcEmbDim);
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
    // Some input sanity checks.
    if (densificationTask.numCopyTasks == 0)
    {
        return;
    }

    // -------------
    // Obtain the real begin and end points we want to densify.
    const size_t embDimToCopyBeginIncl = std::max(densificationTask.globalEmbIdxBeginIncl, m_residentPartitionConfig.getEmbDimBeginIncl());
    const size_t embDimToCopyEndExcl = std::min(densificationTask.globalEmbIdxEndExcl, m_residentPartitionConfig.getEmbDimEndExcl());

    // -------------
    // Prepare the parameters for the kernel.
    DensifyFromResidentKernelParams params;
    {
        // --------
        // Source
        params.d_srcEmbData = m_d_embData.data();
        params.srcNumDocs = m_numDocs;
        params.srcEmbDim = m_residentPartitionConfig.getEmbDim();
        params.srcEmbOffset = embDimToCopyBeginIncl - m_residentPartitionConfig.getEmbDimBeginIncl();

        // --------
        // Destination
        params.d_dstEmbData = densificationTask.d_workingEmbDataset;
        params.dstNumDocs = densificationTask.numDocsToDensify;
        params.dstEmbDim = densificationTask.globalEmbIdxEndExcl - densificationTask.globalEmbIdxBeginIncl;
        params.dstEmbOffset = embDimToCopyBeginIncl - densificationTask.globalEmbIdxBeginIncl;

        // --------
        // Copy tasks
        params.embDimToCopy = embDimToCopyEndExcl - embDimToCopyBeginIncl;
        params.d_copyTasks = densificationTask.d_copyTasks;
        params.numCopyTasks = densificationTask.numCopyTasks;
    }

    // -------------
    // Launch the kernel.
    {
        constexpr size_t kBlockSize = 1024;
        size_t numTasks = params.numCopyTasks * params.embDimToCopy;
        size_t gridSize = (numTasks + kBlockSize - 1) / kBlockSize;
        densifyFromResidentKernel<<<gridSize, kBlockSize, 0, m_cudaStreamRead.get()>>>(params);
        CHECK_CUDA(cudaStreamSynchronize(m_cudaStreamRead.get()));
    }
}
