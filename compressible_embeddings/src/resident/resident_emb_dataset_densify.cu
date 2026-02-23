#include "resident/resident_emb_dataset.hpp"
#include "utils/util.hpp"

/*
Let's explain how this works by an example. Let's say we have:
  - A GlobalEmbDataset with 10M docs, and 1024d embeddings. (10M x 1024).
    - Note "GlobalEmbDataset" is just a "concept", in practice, it is represented by ResidentEmbDataset and CompressibleEmbDataset.
  - We have a CompressibleEmbDataset with 10M docs, and 1024d embeddings, but using residual quantization to compress each emb dim to 2 bits. (10M x 1024)
  - Among those 1024d, [128:512] (384d) are "resident" embeddings, so this ResidentEmbDataset has 10M docs, and 384d embeddings. (10M x 384)
  - Now we assume we have maxNumWorkingDocs = 100k, so we will have a WorkingEmbDataset with 10k docs, and 1024d max embeddings. (100k x 1024)
    - It is important to note that the WorkingEmbDataset is just a "container", exactly how many docs and embedding dims depends on the caller's ask.
  - Now let's say we want to densify desiredDocIdxList = [2, 6, 9, 13], assuming the dims to copy is [192, 768] (576d)
  - And assume in the current WorkingEmbDataset, we have currDocIdxList = [2, 5, 6, 10, 13], we also assume the embedding dim being 576d.
  - So following our explanation in the manager/emb_dataset_manager.cu, we will:
    - Reorder the desiredDocIdxList to [2, 13, 6, 9]
    - Create a list of copy task: copyTasks = [(srcDocIdx=13, dstDocIdx=1), (srcDocIdx=9, dstDocIdx=3)]

So conceptually, we want to perform the following copies:
  - Copy GlobalEmbDataset[13][192:768] to WorkingEmbDataset[1][0:576]
  - Copy GlobalEmbDataset[9][192:768] to WorkingEmbDataset[3][0:576]

However, in practice, since [128:512] are "resident" embeddings, we actually need to split the copy into the following tasks:
  1. Copy ResidentEmbDataset[13][64:384] to WorkingEmbDataset[1][0:320]
  2. Copy ResidentEmbDataset[9][64:384] to WorkingEmbDataset[3][0:320]
  3. Copy CompressibleEmbDataset[13][512:768] to WorkingEmbDataset[1][320:576]
  4. Copy CompressibleEmbDataset[9][512:768] to WorkingEmbDataset[3][320:576]

The kernel in this file, performs (1) and (2), while (3) and (4) are performed in the compressible/res_quant/res_quant_dataset.cu.

It is important to note how those index shifts are calculated.
  - We want [192:768] from the GlobalEmbDataset, but when we write to the WorkingEmbDataset, we want to write to [0:576], not [192:768].
  - The [192:512] resident dims in GlobalEmbDataset, from ResidentEmbDataset's POV, are actually [64:384]

=========== Example 2 ===========

Now let's say the dims to copy is [64:256] (192d) instead, while keeping everything else the same.

We will perform the following copies:
  1. Copy CompressibleEmbDataset[13][64:128] to WorkingEmbDataset[1][0:64]
  2. Copy CompressibleEmbDataset[9][64:128] to WorkingEmbDataset[3][0:64]
  3. Copy ResidentEmbDataset[13][0:128] to WorkingEmbDataset[1][64:192]
  4. Copy ResidentEmbDataset[9][0:128] to WorkingEmbDataset[3][64:192]


*/

struct DensifyFromResidentKernelParams
{
    // Source
    const T_EMB* d_srcEmbData; // 10M x 384 for both examples
    T_DOC_IDX srcNumDocs; // 10M for both examples
    int srcEmbDim; // 384 for both examples
    int srcEmbOffset; // Example 1: 192 - 128 = 64, Example 2: 0

    // Destination
    T_EMB* d_dstEmbData; // 100k x 1024 as the "max", but in reality it is 
                         // 4 x (768 - 192 = 576) for Example 1, 
                         // 4 x (256 -  64 = 192) for Example 2
    int dstNumDocs; // 4 for both examples
    int dstEmbDim; // 576 for Example 1, 192 for Example 2
    int dstEmbOffset; // Example 1: 0, Example 2: 64

    // Copy tasks
    int embDimToCopy; // Example 1: (384 - 64 = 320), Example 2: (256 - 64 = 192)
    CopyTask* d_copyTasks; // [(srcDocIdx=13, dstDocIdx=1), (srcDocIdx=9, dstDocIdx=3)]
    int numCopyTasks; // 2
};

__global__ void densifyFromResidentKernel(DensifyFromResidentKernelParams params)
{
    size_t cudaTaskId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t copyTaskIdx = cudaTaskId / params.embDimToCopy;
    size_t embIdx = cudaTaskId % params.embDimToCopy;

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
