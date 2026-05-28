#include "utils/util.hpp"
#include "worker_split_lock.hpp"

#include <vector>

// Scatters embedding values to non-contiguous rows. Each thread writes one T_EMB value.
static __global__ void kn_scatter(T_EMB* d_dst, const EmbElement* d_elements, int embDim, int numElements)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    d_dst[d_elements[t].rowIdx * embDim + t % embDim] = d_elements[t].val;
}

// Scatters scalar values to non-contiguous (row, scalarIdx) locations.
static __global__ void kn_scatter(float* d_dst, const ScalarElement* d_elements, int numScalars, int numElements)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    d_dst[d_elements[t].rowIdx * numScalars + d_elements[t].scalarIdx] = d_elements[t].val;
}

// Marks deleted rows as dirty so scorers skip them.
static __global__ void kn_setDirty(DirtyBit* d_dirty, const int* d_rowIdx, int numElements)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    d_dirty[d_rowIdx[t]] = DirtyBit::DIRTY;
}

WorkerSplitLock::WorkerSplitLock(int maxNumDocs, int embDim, int numScalars)
    : Worker(maxNumDocs, embDim, numScalars)
{
}

void WorkerSplitLock::upsertDocs(const std::vector<long>& v_docId, const std::vector<std::vector<T_EMB>>& v2_embData)
{
    // --- resolve docId -> rowIdx and build emb elements ---
    std::vector<EmbElement> v_element;
    {
        std::lock_guard<std::mutex> mapLock(m_mapMutex);
        v_element = resolveAndBuildEmbElements(v_docId, v2_embData);
    }

    if (v_element.empty())
    {
        return;
    }

    const int kBlockSize      = 256;
    const int scatterGridSize = (v_element.size() + kBlockSize - 1) / kBlockSize;

    // --- H2D: can overlap with score (no GPU memory touched yet) ---
    CudaDeviceArray<EmbElement> d_element(v_element.size(), "EmbElements");
    CHECK_CUDA(cudaMemcpyAsync(d_element.data(),
                               v_element.data(),
                               v_element.size() * sizeof(EmbElement),
                               cudaMemcpyHostToDevice,
                               m_writeStream.get()));
    CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));

    // --- scatter: exclusive with score kernel ---
    {
        std::lock_guard<std::mutex> gpuLock(m_gpuMutex);
        kn_scatter<<<scatterGridSize, kBlockSize, 0, m_writeStream.get()>>>(m_data.data(),
                                                                            d_element.data(),
                                                                            m_embDim,
                                                                            v_element.size());
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }
}

void WorkerSplitLock::updateScalarData(const std::vector<long>&               v_docId,
                                       const std::vector<std::vector<float>>& v2_scalar)
{
    // --- resolve docId -> rowIdx and build scalar elements ---
    std::vector<ScalarElement> v_scalarElement;
    {
        std::lock_guard<std::mutex> mapLock(m_mapMutex);
        v_scalarElement = resolveScalarElements(v_docId, v2_scalar);
    }

    if (v_scalarElement.empty())
    {
        return;
    }

    const int kBlockSize      = 256;
    const int scatterGridSize = (v_scalarElement.size() + kBlockSize - 1) / kBlockSize;

    // --- H2D: can overlap with score ---
    CudaDeviceArray<ScalarElement> d_scalarElement(v_scalarElement.size(), "scalarElements");
    CHECK_CUDA(cudaMemcpyAsync(d_scalarElement.data(),
                               v_scalarElement.data(),
                               v_scalarElement.size() * sizeof(ScalarElement),
                               cudaMemcpyHostToDevice,
                               m_writeStream.get()));
    CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));

    // --- scatter: exclusive with score kernel ---
    {
        std::lock_guard<std::mutex> gpuLock(m_gpuMutex);
        kn_scatter<<<scatterGridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_scalars.data(),
                                                                            d_scalarElement.data(),
                                                                            m_numScalars,
                                                                            v_scalarElement.size());
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }
}

void WorkerSplitLock::deleteDocs(const std::vector<long>& v_docId)
{
    // --- update maps, collect deleted rowIdxs ---
    std::vector<int> v_deletedRowIdx;
    {
        std::lock_guard<std::mutex> mapLock(m_mapMutex);
        v_deletedRowIdx = resolveDeletedRowIdxs(v_docId);
    }

    if (v_deletedRowIdx.empty())
    {
        return;
    }

    const int kBlockSize = 256;
    const int gridSize   = (v_deletedRowIdx.size() + kBlockSize - 1) / kBlockSize;

    // --- H2D: can overlap with score ---
    CudaDeviceArray<int> d_deletedRowIdx(v_deletedRowIdx.size(), "deletedRowIdx");
    CHECK_CUDA(cudaMemcpyAsync(d_deletedRowIdx.data(),
                               v_deletedRowIdx.data(),
                               v_deletedRowIdx.size() * sizeof(int),
                               cudaMemcpyHostToDevice,
                               m_writeStream.get()));
    CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));

    // --- set dirty=1: exclusive with score kernel ---
    // Sync inside the lock so m_d_dirty is fully written before any scorer
    // can acquire m_gpuMutex and launch kn_score.
    {
        std::lock_guard<std::mutex> gpuLock(m_gpuMutex);
        kn_setDirty<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                      d_deletedRowIdx.data(),
                                                                      v_deletedRowIdx.size());
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }
}

void WorkerSplitLock::score(const std::vector<T_EMB>& v_reqEmb,
                            const std::vector<int>&   v_targetRowIdx,
                            int                       targetScalarIdx)
{
    std::lock_guard<std::mutex> gpuLock(m_gpuMutex);
    scoreImpl(v_reqEmb, v_targetRowIdx, targetScalarIdx);
}
