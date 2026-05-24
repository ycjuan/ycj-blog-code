#include "utils/util.hpp"
#include "worker_naive.hpp"

#include <tuple>
#include <vector>

struct ScalarElement
{
    int rowIdx;
    float val;
};

__global__ void kn_scatterScalar(float* dst, const ScalarElement* elements, int numElements)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    dst[elements[t].rowIdx] = elements[t].val;
}

__global__ void kn_setDirty(char* dirty, const int* rowIdxs, int numElements)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    dirty[rowIdxs[t]] = 1;
}

__global__ void kn_scatter(T_EMB* dst, const CopyElement* elements, int embDim, int numElements)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    dst[elements[t].rowIdx * embDim + t % embDim] = elements[t].val;
}

WorkerNaive::WorkerNaive(int maxNumDocs, int embDim)
    : Worker(maxNumDocs, embDim)
{
}

void WorkerNaive::updateScalarData(const std::vector<long>& v_docId, const std::vector<float>& v_scalar)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    // --- resolve docId -> rowIdx and build scalar elements ---
    std::vector<ScalarElement> v_scalarElement;
    {
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

    // --- H2D: scalar data, scatter ---
    {
        CudaDeviceArray<ScalarElement> d_scalarElement(v_scalarElement.size(), "scalarElements");
        CHECK_CUDA(cudaMemcpyAsync(d_scalarElement.data(),
                                   v_scalarElement.data(),
                                   v_scalarElement.size() * sizeof(ScalarElement),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
        const int kBlockSize = 256;
        const int gridSize = (v_scalarElement.size() + kBlockSize - 1) / kBlockSize;
        kn_scatterScalar<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_scalars.data(),
                                                                           d_scalarElement.data(),
                                                                           v_scalarElement.size());
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }
}

void WorkerNaive::upsertDocs(const std::vector<long>& v_docId, const std::vector<std::vector<T_EMB>>& v2_embData)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    // --- resolve docId -> rowIdx and build copy elements ---
    std::vector<CopyElement> v_element;
    {
        auto [v_rowIdx, v_elem] = resolveAndBuildCopyElements(v_docId, v2_embData);
        v_element = std::move(v_elem);
    }

    if (v_element.empty())
    {
        return;
    }

    // --- H2D: emb data, scatter ---
    {
        CudaDeviceArray<CopyElement> d_elements(v_element.size(), "CopyElements");
        CHECK_CUDA(cudaMemcpyAsync(d_elements.data(),
                                   v_element.data(),
                                   v_element.size() * sizeof(CopyElement),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
        const int kBlockSize = 256;
        const int gridSize = (v_element.size() + kBlockSize - 1) / kBlockSize;
        kn_scatter<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_data.data(),
                                                                     d_elements.data(),
                                                                     m_embDim,
                                                                     v_element.size());
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }
}

void WorkerNaive::deleteDocs(const std::vector<long>& v_docId)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    // --- update maps, collect deleted rowIdxs ---
    std::vector<int> v_deletedRowIdx;
    {
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

    // --- H2D: deleted rowIdxs, set dirty=1 ---
    {
        CudaDeviceArray<int> d_deletedRowIdx(v_deletedRowIdx.size(), "deletedRowIdx");
        CHECK_CUDA(cudaMemcpyAsync(d_deletedRowIdx.data(),
                                   v_deletedRowIdx.data(),
                                   v_deletedRowIdx.size() * sizeof(int),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
        const int kBlockSize = 256;
        const int gridSize = (v_deletedRowIdx.size() + kBlockSize - 1) / kBlockSize;
        kn_setDirty<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                      d_deletedRowIdx.data(),
                                                                      v_deletedRowIdx.size());
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }
}

void WorkerNaive::score(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdx)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    scoreImpl(v_reqEmb, v_targetRowIdx);
}
