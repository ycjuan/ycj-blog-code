#include "utils/util.hpp"
#include "worker_naive.hpp"

#include <tuple>
#include <vector>

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

WorkerNaive::WorkerNaive(int maxNumDocs, int embDim, int numScalars)
    : Worker(maxNumDocs, embDim, numScalars)
{
}

void WorkerNaive::updateScalarData(const std::vector<long>& v_docId, const std::vector<std::vector<float>>& v2_scalar)
{
    // --- lock: serialize all GPU ops and map access ---
    std::lock_guard<std::mutex> lock(m_mutex);

    // --- resolve docId -> rowIdx and build scalar elements ---
    std::vector<ScalarElement> v_scalarElement = resolveScalarElements(v_docId, v2_scalar);

    if (v_scalarElement.empty())
    {
        return;
    }

    // --- H2D: scalar data, scatter ---
    {
        CudaDeviceArray<ScalarElement> d_scalarElement(v_scalarElement.size(), "scalarElements");
        CHECK_CUDA(cudaMemcpyAsync(d_scalarElement.data(),
                                   v_scalarElement.data(),
                                   v_scalarElement.size() * sizeof(ScalarElement),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
        const int kBlockSize = 256;
        const int gridSize   = (v_scalarElement.size() + kBlockSize - 1) / kBlockSize;
        kn_scatter<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_scalars.data(),
                                                                     d_scalarElement.data(),
                                                                     m_numScalars,
                                                                     v_scalarElement.size());
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }
}

void WorkerNaive::upsertDocs(const std::vector<long>& v_docId, const std::vector<std::vector<T_EMB>>& v2_embData)
{
    // --- lock: serialize all GPU ops and map access ---
    std::lock_guard<std::mutex> lock(m_mutex);

    // --- resolve docId -> rowIdx and build copy elements ---
    std::vector<EmbElement> v_element = resolveAndBuildEmbElements(v_docId, v2_embData);

    if (v_element.empty())
    {
        return;
    }

    // --- H2D: emb data, scatter ---
    {
        CudaDeviceArray<EmbElement> d_elements(v_element.size(), "EmbElements");
        CHECK_CUDA(cudaMemcpyAsync(d_elements.data(),
                                   v_element.data(),
                                   v_element.size() * sizeof(EmbElement),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
        const int kBlockSize = 256;
        const int gridSize   = (v_element.size() + kBlockSize - 1) / kBlockSize;
        kn_scatter<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_data.data(),
                                                                     d_elements.data(),
                                                                     m_embDim,
                                                                     v_element.size());
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }
}

void WorkerNaive::deleteDocs(const std::vector<long>& v_docId)
{
    // --- lock: serialize all GPU ops and map access ---
    std::lock_guard<std::mutex> lock(m_mutex);

    // --- update maps, collect deleted rowIdxs ---
    std::vector<int> v_deletedRowIdx = resolveDeletedRowIdxs(v_docId);

    if (v_deletedRowIdx.empty())
    {
        return;
    }

    // --- H2D: deleted rowIdxs, set dirty=1 ---
    {
        CudaDeviceArray<int> d_deletedRowIdx(v_deletedRowIdx.size(), "deletedRowIdx");
        CHECK_CUDA(cudaMemcpyAsync(d_deletedRowIdx.data(),
                                   v_deletedRowIdx.data(),
                                   v_deletedRowIdx.size() * sizeof(int),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
        const int kBlockSize = 256;
        const int gridSize   = (v_deletedRowIdx.size() + kBlockSize - 1) / kBlockSize;
        kn_setDirty<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                      d_deletedRowIdx.data(),
                                                                      v_deletedRowIdx.size());
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }
}

void WorkerNaive::score(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdx, int targetScalarIdx)
{
    // --- lock: serialize all GPU ops and map access ---
    std::lock_guard<std::mutex> lock(m_mutex);
    scoreImpl(v_reqEmb, v_targetRowIdx, targetScalarIdx);
}
