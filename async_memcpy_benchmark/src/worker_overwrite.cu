#include "utils/util.hpp"
#include "worker_overwrite.hpp"

#include <tuple>
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

// Sets d_dirty[d_rowIdx[t]] = val. Used for delete and scalar dirty-bit fencing.
static __global__ void kn_setDirty(DirtyBit* d_dirty, const int* d_rowIdx, int numElements, DirtyBit val)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    d_dirty[d_rowIdx[t]] = val;
}

// Sets dirty bit for one row per doc using the EmbElement array. Each doc occupies
// a contiguous block of embDim elements, so element[t * embDim] is the first element
// of doc t and carries the correct rowIdx.
static __global__ void kn_setDirty(DirtyBit*         d_dirty,
                                   const EmbElement* d_elements,
                                   int               embDim,
                                   int               numDocs,
                                   DirtyBit          val)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numDocs)
    {
        return;
    }
    d_dirty[d_elements[t * embDim].rowIdx] = val;
}

WorkerOverwrite::WorkerOverwrite(int maxNumDocs, int embDim, int numScalars)
    : Worker(maxNumDocs, embDim, numScalars)
{
}

void WorkerOverwrite::upsertDocs(const std::vector<long>& v_docId, const std::vector<std::vector<T_EMB>>& v2_embData)
{
    // --- resolve docId -> rowIdx and build copy elements ---
    std::vector<EmbElement> v_element;
    {
        // --- lock: protect docId<>rowIdx map ---
        std::lock_guard<std::mutex> lock(m_writeMutex);
        v_element = resolveAndBuildEmbElements(v_docId, v2_embData);
    }

    if (v_element.empty())
    {
        return;
    }

    const int kBlockSize = 256;
    const int numDocs    = v_docId.size();

    // d_element lives for the whole function: the dirty-bit kernels index into it
    // by doc (stride embDim), so it must outlive both dirty-set and dirty-clear launches.
    CudaDeviceArray<EmbElement> d_element(v_element.size(), "EmbElements");
    int                         dirtyGridSize   = (numDocs + kBlockSize - 1) / kBlockSize;
    int                         scatterGridSize = (v_element.size() + kBlockSize - 1) / kBlockSize;

    // --- H2D: emb data (can happen before dirty=1 is set) ---
    {
        CHECK_CUDA(cudaMemcpyAsync(d_element.data(),
                                   v_element.data(),
                                   v_element.size() * sizeof(EmbElement),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }

    // --- set dirty=1 before scatter ---
    {
        // --- lock: protect dirty bits from concurrent score reads ---
        std::lock_guard<std::mutex> lock(m_readMutex);
        kn_setDirty<<<dirtyGridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                           d_element.data(),
                                                                           m_embDim,
                                                                           numDocs,
                                                                           DirtyBit::DIRTY);
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }

    // --- scatter emb data ---
    {
        kn_scatter<<<scatterGridSize, kBlockSize, 0, m_writeStream.get()>>>(m_data.data(),
                                                                            d_element.data(),
                                                                            m_embDim,
                                                                            v_element.size());
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }

    // --- clear dirty=0 after scatter ---
    {
        // --- lock: protect dirty bits from concurrent score reads ---
        std::lock_guard<std::mutex> lock(m_readMutex);
        kn_setDirty<<<dirtyGridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                           d_element.data(),
                                                                           m_embDim,
                                                                           numDocs,
                                                                           DirtyBit::CLEAN);
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }
}

void WorkerOverwrite::updateScalarData(const std::vector<long>&               v_docId,
                                       const std::vector<std::vector<float>>& v2_scalar)
{
    // --- resolve docId -> rowIdx and build scalar elements ---
    std::vector<ScalarElement> v_scalarElement;
    {
        // --- lock: protect docId<>rowIdx map ---
        std::lock_guard<std::mutex> lock(m_writeMutex);
        v_scalarElement = resolveScalarElements(v_docId, v2_scalar);
    }

    if (v_scalarElement.empty())
    {
        return;
    }

    // resolveScalarElements emits numScalars elements per doc, all with the same rowIdx.
    // Extract one rowIdx per doc by scanning for group boundaries.
    std::vector<int> v_dirtyRowIdx;
    for (int i = 0; i < (int)v_scalarElement.size();)
    {
        int rowIdx = v_scalarElement[i].rowIdx;
        v_dirtyRowIdx.push_back(rowIdx);
        while (i < (int)v_scalarElement.size() && v_scalarElement[i].rowIdx == rowIdx)
            i++;
    }

    const int kBlockSize  = 256;
    const int numElements = v_scalarElement.size();
    const int numDocs     = v_dirtyRowIdx.size();

    // d_scalarElement and d_dirtyRowIdx at function scope — shared across sections.
    CudaDeviceArray<ScalarElement> d_scalarElement(numElements, "scalarElements");
    CudaDeviceArray<int>           d_dirtyRowIdx(numDocs, "dirtyRowIdx");
    int                            scatterGridSize = (numElements + kBlockSize - 1) / kBlockSize;
    int                            dirtyGridSize   = (numDocs + kBlockSize - 1) / kBlockSize;

    // --- H2D: scalar data and dirty rowIdxs (can happen before dirty=1 is set) ---
    {
        CHECK_CUDA(cudaMemcpyAsync(d_scalarElement.data(),
                                   v_scalarElement.data(),
                                   numElements * sizeof(ScalarElement),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
        CHECK_CUDA(cudaMemcpyAsync(d_dirtyRowIdx.data(),
                                   v_dirtyRowIdx.data(),
                                   numDocs * sizeof(int),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
    }

    // --- set dirty=1 before scatter ---
    {
        // --- lock: protect dirty bits from concurrent score reads ---
        std::lock_guard<std::mutex> lock(m_readMutex);
        kn_setDirty<<<dirtyGridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                           d_dirtyRowIdx.data(),
                                                                           numDocs,
                                                                           DirtyBit::DIRTY);
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }

    // --- scatter scalar data ---
    {
        kn_scatter<<<scatterGridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_scalars.data(),
                                                                            d_scalarElement.data(),
                                                                            m_numScalars,
                                                                            numElements);
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }

    // --- clear dirty=0 after scatter ---
    {
        // --- lock: protect dirty bits from concurrent score reads ---
        std::lock_guard<std::mutex> lock(m_readMutex);
        kn_setDirty<<<dirtyGridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                           d_dirtyRowIdx.data(),
                                                                           numDocs,
                                                                           DirtyBit::CLEAN);
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }
}

void WorkerOverwrite::deleteDocs(const std::vector<long>& v_docId)
{
    // --- update maps, collect deleted rowIdxs ---
    std::vector<int> v_deletedRowIdx;
    {
        // --- lock: protect docId<>rowIdx map ---
        std::lock_guard<std::mutex> lock(m_writeMutex);
        v_deletedRowIdx = resolveDeletedRowIdxs(v_docId);
    }

    if (v_deletedRowIdx.empty())
    {
        return;
    }

    const int kBlockSize = 256;
    const int gridSize   = (v_deletedRowIdx.size() + kBlockSize - 1) / kBlockSize;

    // --- H2D: deleted rowIdxs, set dirty=1 ---
    {
        CudaDeviceArray<int> d_deletedRowIdx(v_deletedRowIdx.size(), "deletedRowIdx");
        CHECK_CUDA(cudaMemcpyAsync(d_deletedRowIdx.data(),
                                   v_deletedRowIdx.data(),
                                   v_deletedRowIdx.size() * sizeof(int),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
        {
            // --- lock: protect dirty bits from concurrent score reads ---
            std::lock_guard<std::mutex> lock(m_readMutex);
            kn_setDirty<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                          d_deletedRowIdx.data(),
                                                                          v_deletedRowIdx.size(),
                                                                          DirtyBit::DIRTY);
        }
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }
}

void WorkerOverwrite::score(const std::vector<T_EMB>& v_reqEmb,
                            const std::vector<int>&   v_targetRowIdx,
                            int                       targetScalarIdx)
{
    // --- lock: protect dirty bits from concurrent write ops ---
    std::lock_guard<std::mutex> lock(m_readMutex);
    scoreImpl(v_reqEmb, v_targetRowIdx, targetScalarIdx);
}
