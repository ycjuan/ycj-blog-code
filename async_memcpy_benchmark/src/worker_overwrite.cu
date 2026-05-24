#include "utils/util.hpp"
#include "worker_overwrite.hpp"

#include <tuple>
#include <vector>

struct ScalarElement
{
    int rowIdx;
    float val;
};

__global__ void kn_scatter(T_EMB* dst, const CopyElement* elements, int embDim, int numElements)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    dst[elements[t].rowIdx * embDim + t % embDim] = elements[t].val;
}

__global__ void kn_scatter(float* dst, const ScalarElement* elements, int numElements)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    dst[elements[t].rowIdx] = elements[t].val;
}

__global__ void kn_setDirty(char* dirty, const int* rowIdxs, int numElements, char val)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    dirty[rowIdxs[t]] = val;
}

__global__ void kn_setDirty(char* dirty, const CopyElement* elements, int embDim, int numDocs, char val)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numDocs)
    {
        return;
    }
    dirty[elements[t * embDim].rowIdx] = val;
}

__global__ void kn_setDirty(char* dirty, const ScalarElement* elements, int numDocs, char val)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numDocs)
    {
        return;
    }
    dirty[elements[t].rowIdx] = val;
}

WorkerOverwrite::WorkerOverwrite(int maxNumDocs, int embDim)
    : Worker(maxNumDocs, embDim)
{
}

void WorkerOverwrite::upsertDocs(const std::vector<long>& v_docId, const std::vector<std::vector<T_EMB>>& v2_embData)
{
    // --- resolve docId -> rowIdx and build copy elements ---
    std::vector<CopyElement> v_element;
    {
        // --- lock: protect docId<>rowIdx map ---
        std::lock_guard<std::mutex> lock(m_writeMutex);
        v_element = resolveAndBuildCopyElements(v_docId, v2_embData);
    }

    if (v_element.empty())
    {
        return;
    }

    const int kBlockSize = 256;
    const int numDocs = v_docId.size();

    // d_element at function scope — shared across sections, must
    // outlive all kernel launches until the final sync in each section.
    CudaDeviceArray<CopyElement> d_element(v_element.size(), "CopyElements");
    int dirtyGridSize = (numDocs + kBlockSize - 1) / kBlockSize;
    int scatterGridSize = (v_element.size() + kBlockSize - 1) / kBlockSize;

    // --- H2D: emb data (can happen before dirty=1 is set) ---
    {
        CHECK_CUDA(cudaMemcpyAsync(d_element.data(),
                                   v_element.data(),
                                   v_element.size() * sizeof(CopyElement),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
    }

    // --- set dirty=1 before scatter ---
    {
        // --- lock: protect dirty bits from concurrent score reads ---
        std::lock_guard<std::mutex> lock(m_readMutex);
        kn_setDirty<<<dirtyGridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                           d_element.data(),
                                                                           m_embDim,
                                                                           numDocs,
                                                                           1);
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
                                                                           0);
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }
}

void WorkerOverwrite::updateScalarData(const std::vector<long>& v_docId, const std::vector<float>& v_scalar)
{
    // --- resolve docId -> rowIdx and build scalar elements ---
    std::vector<ScalarElement> v_scalarElement;
    {
        // --- lock: protect docId<>rowIdx map ---
        std::lock_guard<std::mutex> lock(m_writeMutex);
        v_scalarElement.reserve(v_docId.size());

        for (int i = 0; i < (int)v_docId.size(); i++)
        {
            auto it = m_docId2rowIdx.find(v_docId[i]);
            if (it == m_docId2rowIdx.end())
            {
                continue;
            }
            v_scalarElement.push_back({ it->second, v_scalar[i] });
        }
    }

    if (v_scalarElement.empty())
    {
        return;
    }

    const int kBlockSize = 256;
    const int numDocs = v_scalarElement.size();

    // d_scalarElement at function scope — shared across sections.
    CudaDeviceArray<ScalarElement> d_scalarElement(numDocs, "scalarElements");
    int dirtyGridSize = (numDocs + kBlockSize - 1) / kBlockSize;

    // --- H2D: scalar data (can happen before dirty=1 is set) ---
    {
        CHECK_CUDA(cudaMemcpyAsync(d_scalarElement.data(),
                                   v_scalarElement.data(),
                                   numDocs * sizeof(ScalarElement),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
    }

    // --- set dirty=1 before scatter ---
    {
        // --- lock: protect dirty bits from concurrent score reads ---
        std::lock_guard<std::mutex> lock(m_readMutex);
        kn_setDirty<<<dirtyGridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                           d_scalarElement.data(),
                                                                           numDocs,
                                                                           1);
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }

    // --- scatter scalar data ---
    {
        kn_scatter<<<dirtyGridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_scalars.data(),
                                                                          d_scalarElement.data(),
                                                                          numDocs);
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }

    // --- clear dirty=0 after scatter ---
    {
        std::lock_guard<std::mutex> lock(m_readMutex);
        kn_setDirty<<<dirtyGridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                           d_scalarElement.data(),
                                                                           numDocs,
                                                                           0);
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
    const int gridSize = (v_deletedRowIdx.size() + kBlockSize - 1) / kBlockSize;

    // --- H2D: deleted rowIdxs, set dirty=1 ---
    {
        CudaDeviceArray<int> d_rowIdx(v_deletedRowIdx.size(), "deletedRowIdx");
        CHECK_CUDA(cudaMemcpyAsync(d_rowIdx.data(),
                                   v_deletedRowIdx.data(),
                                   v_deletedRowIdx.size() * sizeof(int),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
        {
            // --- lock: protect dirty bits from concurrent score reads ---
            std::lock_guard<std::mutex> lock(m_readMutex);
            kn_setDirty<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                          d_rowIdx.data(),
                                                                          v_deletedRowIdx.size(),
                                                                          1);
        }
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }
}

void WorkerOverwrite::score(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdx)
{
    // --- lock: protect dirty bits from concurrent write ops ---
    std::lock_guard<std::mutex> lock(m_readMutex);
    scoreImpl(v_reqEmb, v_targetRowIdx);
}
