#include "utils/util.hpp"
#include "worker_overwrite.hpp"

#include <vector>

struct CopyElement
{
    int rowIdx;
    T_EMB val;
};

struct ScalarElement
{
    int rowIdx;
    float val;
};

__global__ void scatterKernel(T_EMB* dst, const CopyElement* elements, int embDim, int numElements)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    dst[elements[t].rowIdx * embDim + t % embDim] = elements[t].val;
}

__global__ void scatterScalarKernel(float* dst, const ScalarElement* elements, int numElements)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    dst[elements[t].rowIdx] = elements[t].val;
}

__global__ void setDirtyKernel(char* dirty, const int* rowIdxs, int numElements, char val)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    dirty[rowIdxs[t]] = val;
}

WorkerOverwrite::WorkerOverwrite(int maxNumDocs, int embDim)
    : Worker(maxNumDocs, embDim)
{
}

void WorkerOverwrite::upsertDocs(const std::vector<long>& v_docId, const std::vector<std::vector<T_EMB>>& v2_embData)
{
    // --- resolve docId -> rowIdx and build copy elements ---
    std::vector<int> v_rowIdx;
    std::vector<CopyElement> v_element;
    {
        std::lock_guard<std::mutex> lock(m_writeMutex);
        v_rowIdx.reserve(v_docId.size());
        v_element.reserve(v_docId.size() * m_embDim);

        for (int i = 0; i < (int)v_docId.size(); i++)
        {
            auto it = m_docId2rowIdx.find(v_docId[i]);
            int rowIdx;
            if (it == m_docId2rowIdx.end())
            {
                if (!m_emptyRowIdxSet.empty())
                {
                    rowIdx = *m_emptyRowIdxSet.begin();
                    m_emptyRowIdxSet.erase(m_emptyRowIdxSet.begin());
                }
                else
                {
                    rowIdx = m_headRowIdx++;
                }
                m_docId2rowIdx[v_docId[i]] = rowIdx;
                m_rowIdx2DocId[rowIdx] = v_docId[i];
            }
            else
            {
                rowIdx = it->second;
            }
            v_rowIdx.push_back(rowIdx);
            for (int j = 0; j < m_embDim; j++)
            {
                v_element.push_back({ rowIdx, v2_embData[i][j] });
            }
        }
    }

    if (v_element.empty())
    {
        return;
    }

    const int kBlockSize = 256;

    // d_rowIdx is declared here (function scope) because it is shared across
    // the two dirty-bit sections below and must outlive both kernel launches.
    CudaDeviceArray<int> d_rowIdx(v_rowIdx.size(), "rowIdx");
    int dirtyGridSize = (v_rowIdx.size() + kBlockSize - 1) / kBlockSize;

    // --- H2D: rowIdxs ---
    {
        CHECK_CUDA(cudaMemcpyAsync(d_rowIdx.data(),
                                   v_rowIdx.data(),
                                   v_rowIdx.size() * sizeof(int),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
    }

    // --- set dirty=1 before writing emb data ---
    {
        std::lock_guard<std::mutex> lock(m_readMutex);
        setDirtyKernel<<<dirtyGridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                              d_rowIdx.data(),
                                                                              v_rowIdx.size(),
                                                                              1);
    }

    // --- H2D: emb data, scatter ---
    {
        CudaDeviceArray<CopyElement> d_element(v_element.size(), "CopyElements");
        CHECK_CUDA(cudaMemcpyAsync(d_element.data(),
                                   v_element.data(),
                                   v_element.size() * sizeof(CopyElement),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
        int scatterGridSize = (v_element.size() + kBlockSize - 1) / kBlockSize;
        scatterKernel<<<scatterGridSize, kBlockSize, 0, m_writeStream.get()>>>(m_data.data(),
                                                                               d_element.data(),
                                                                               m_embDim,
                                                                               v_element.size());
    }

    // --- clear dirty=0 after scatter (ordered on same stream) ---
    {
        std::lock_guard<std::mutex> lock(m_readMutex);
        setDirtyKernel<<<dirtyGridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                              d_rowIdx.data(),
                                                                              v_rowIdx.size(),
                                                                              0);
    }

    // --- sync outside lock so score() can run concurrently on readStream ---
    CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
}

void WorkerOverwrite::updateScalarData(const std::vector<long>& v_docId, const std::vector<float>& v_scalar)
{
    // --- resolve docId -> rowIdx and build scalar elements ---
    std::vector<int> v_rowIdx;
    std::vector<ScalarElement> v_scalarElement;
    {
        std::lock_guard<std::mutex> lock(m_writeMutex);
        v_rowIdx.reserve(v_docId.size());
        v_scalarElement.reserve(v_docId.size());

        for (int i = 0; i < (int)v_docId.size(); i++)
        {
            auto it = m_docId2rowIdx.find(v_docId[i]);
            if (it == m_docId2rowIdx.end())
            {
                continue;
            }
            v_rowIdx.push_back(it->second);
            v_scalarElement.push_back({ it->second, v_scalar[i] });
        }
    }

    if (v_scalarElement.empty())
    {
        return;
    }

    const int kBlockSize = 256;

    // d_rowIdx declared at function scope — shared across dirty-bit sections.
    CudaDeviceArray<int> d_rowIdx(v_rowIdx.size(), "rowIdx");
    int dirtyGridSize = (v_rowIdx.size() + kBlockSize - 1) / kBlockSize;

    // --- H2D: rowIdxs ---
    {
        CHECK_CUDA(cudaMemcpyAsync(d_rowIdx.data(),
                                   v_rowIdx.data(),
                                   v_rowIdx.size() * sizeof(int),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
    }

    // --- set dirty=1 before writing scalar data ---
    {
        std::lock_guard<std::mutex> lock(m_readMutex);
        setDirtyKernel<<<dirtyGridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                              d_rowIdx.data(),
                                                                              v_rowIdx.size(),
                                                                              1);
    }

    // --- H2D: scalar data, scatter ---
    {
        CudaDeviceArray<ScalarElement> d_scalarElement(v_scalarElement.size(), "scalarElements");
        CHECK_CUDA(cudaMemcpyAsync(d_scalarElement.data(),
                                   v_scalarElement.data(),
                                   v_scalarElement.size() * sizeof(ScalarElement),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
        int scatterGridSize = (v_scalarElement.size() + kBlockSize - 1) / kBlockSize;
        scatterScalarKernel<<<scatterGridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_scalars.data(),
                                                                                     d_scalarElement.data(),
                                                                                     v_scalarElement.size());
    }

    // --- clear dirty=0 after scatter (ordered on same stream) ---
    {
        std::lock_guard<std::mutex> lock(m_readMutex);
        setDirtyKernel<<<dirtyGridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                              d_rowIdx.data(),
                                                                              v_rowIdx.size(),
                                                                              0);
    }

    CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
}

void WorkerOverwrite::deleteDocs(const std::vector<long>& v_docId)
{
    // --- update maps, collect deleted rowIdxs ---
    std::vector<int> v_deletedRowIdx;
    {
        std::lock_guard<std::mutex> lock(m_writeMutex);
        v_deletedRowIdx.reserve(v_docId.size());

        for (long docId : v_docId)
        {
            auto it = m_docId2rowIdx.find(docId);
            if (it == m_docId2rowIdx.end())
            {
                continue;
            }
            int rowIdx = it->second;
            m_docId2rowIdx.erase(it);
            m_rowIdx2DocId[rowIdx] = -1;
            m_emptyRowIdxSet.insert(rowIdx);
            v_deletedRowIdx.push_back(rowIdx);
        }
    }

    if (v_deletedRowIdx.empty())
    {
        return;
    }

    const int kBlockSize = 256;
    const int gridSize = (v_deletedRowIdx.size() + kBlockSize - 1) / kBlockSize;

    // --- H2D: deleted rowIdxs, set dirty=1 ---
    {
        CudaDeviceArray<int> d_deletedRowIdx(v_deletedRowIdx.size(), "deletedRowIdx");
        CHECK_CUDA(cudaMemcpyAsync(d_deletedRowIdx.data(),
                                   v_deletedRowIdx.data(),
                                   v_deletedRowIdx.size() * sizeof(int),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
        {
            std::lock_guard<std::mutex> lock(m_readMutex);
            setDirtyKernel<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                             d_deletedRowIdx.data(),
                                                                             v_deletedRowIdx.size(),
                                                                             1);
        }
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }
}

void WorkerOverwrite::score(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdx)
{
    std::lock_guard<std::mutex> lock(m_readMutex);
    scoreImpl(v_reqEmb, v_targetRowIdx);
}
