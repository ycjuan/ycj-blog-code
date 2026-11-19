#include "backends.hpp"
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>

#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t e = (call);                                                                                        \
        if (e != cudaSuccess)                                                                                          \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e));                                   \
    } while (0)

#define CHECK_CUBLAS(call)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        cublasStatus_t s = (call);                                                                                     \
        if (s != CUBLAS_STATUS_SUCCESS)                                                                                \
            throw std::runtime_error("cuBLAS error: " + std::to_string(static_cast<int>(s)));                          \
    } while (0)

namespace
{

std::vector<float> loadBin(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("Cannot open: " + path);
    size_t bytes = f.tellg();
    f.seekg(0);
    std::vector<float> d(bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(d.data()), bytes);
    return d;
}

struct GpuBuf
{
    float* ptr = nullptr;
    size_t n   = 0;

    GpuBuf() = default;
    explicit GpuBuf(size_t n_)
        : n(n_)
    {
        CHECK_CUDA(cudaMalloc(&ptr, n * sizeof(float)));
    }
    ~GpuBuf()
    {
        if (ptr)
            cudaFree(ptr);
    }

    GpuBuf(const GpuBuf&)            = delete;
    GpuBuf& operator=(const GpuBuf&) = delete;
    GpuBuf(GpuBuf&& o) noexcept
        : ptr(o.ptr)
        , n(o.n)
    {
        o.ptr = nullptr;
        o.n   = 0;
    }
    GpuBuf& operator=(GpuBuf&& o) noexcept
    {
        if (ptr)
            cudaFree(ptr);
        ptr   = o.ptr;
        n     = o.n;
        o.ptr = nullptr;
        o.n   = 0;
        return *this;
    }

    void upload(const std::vector<float>& h)
    {
        CHECK_CUDA(cudaMemcpy(ptr, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    std::vector<float> download() const
    {
        std::vector<float> h(n);
        CHECK_CUDA(cudaMemcpy(h.data(), ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
        return h;
    }
};

__global__ void addBiasColMajor(float* A, const float* b, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols)
        A[col * rows + row] += b[row];
}

__global__ void reluInPlace(float* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = fmaxf(0.0f, x[i]);
}

// Combines the per-query and per-doc layer-1 projections into h1[H1, R*N] (col-major),
// where column c = r * N + n pairs query r with doc n. Equivalent to the broadcast add
// query_proj.unsqueeze(1) + docs_proj.unsqueeze(0) in export.py's forward().
__global__ void addBroadcastQueryDoc(float* out, const float* query_proj, const float* docs_proj, int H1, int R, int N)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (h < H1 && c < R * N)
    {
        int r           = c / N;
        int n           = c % N;
        out[h + c * H1] = query_proj[h + r * H1] + docs_proj[h + n * H1];
    }
}

__global__ void sigmoidInPlace(float* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = 1.0f / (1.0f + expf(-x[i]));
}

void gemm(cublasHandle_t h, const float* W, int out_dim, int in_dim, const float* X, int N, float* C)
{
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(
        cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, out_dim, N, in_dim, &alpha, W, in_dim, X, in_dim, &beta, C, out_dim));
}

} // namespace

struct CudaBackend : InferBackend
{
    cublasHandle_t   handle;
    std::vector<int> hidden_sizes;
    int              query_dim, doc_dim, num_heads;

    GpuBuf              d_w1_query, d_b1, d_w1_doc;
    std::vector<GpuBuf> d_w_hidden, d_b_hidden;
    GpuBuf              d_w_out, d_b_out;

    CudaBackend(const Paths& paths, const Input& shape_hint)
        : query_dim(shape_hint.query_dim)
        , doc_dim(shape_hint.doc_dim)
        , num_heads(shape_hint.num_heads)
    {
        CHECK_CUBLAS(cublasCreate(&handle));

        const std::string& wdir = paths.weights_dir;

        auto upload = [&](const std::string& name)
        {
            auto   h = loadBin(wdir + name);
            GpuBuf buf(h.size());
            buf.upload(h);
            return buf;
        };

        d_w1_query = upload("w1_query.bin");
        d_b1       = upload("b1.bin");
        d_w1_doc   = upload("w1_doc.bin");

        hidden_sizes.push_back(static_cast<int>(d_b1.n));
        for (int i = 0;; ++i)
        {
            std::ifstream f(wdir + "w_hidden_" + std::to_string(i) + ".bin", std::ios::binary);
            if (!f)
                break;
            f.seekg(0, std::ios::end);
            int prev = hidden_sizes.back();
            hidden_sizes.push_back(static_cast<int>(f.tellg() / sizeof(float)) / prev);
            d_w_hidden.push_back(upload("w_hidden_" + std::to_string(i) + ".bin"));
            d_b_hidden.push_back(upload("b_hidden_" + std::to_string(i) + ".bin"));
        }

        d_w_out = upload("w_out.bin");
        d_b_out = upload("b_out.bin");
    }

    ~CudaBackend() { cublasDestroy(handle); }

    bool supports_device_infer() const override { return true; }

    // d_scores output is col-major [num_heads x num_docs] (native cuBLAS layout).
    void infer_device(const float* d_query,
                      const float* d_docs,
                      float*       d_scores,
                      int          num_reqs,
                      int          query_dim_,
                      int          doc_dim_,
                      int          num_docs,
                      int          num_heads_) override
    {
        const int H1   = hidden_sizes[0];
        const int maxH = *std::max_element(hidden_sizes.begin(), hidden_sizes.end());
        const int R    = num_reqs;
        const int N    = num_docs;
        const int cols = R * N; // one column per (query, doc) pair

        GpuBuf d_query_proj(H1 * R);
        GpuBuf d_docs_proj(H1 * N);
        GpuBuf d_act_a(maxH * cols);
        GpuBuf d_act_b(maxH * cols);

        // query_proj [H1 x R] = W1_query @ query^T, then + b1 broadcast over R
        gemm(handle, d_w1_query.ptr, H1, query_dim_, d_query, R, d_query_proj.ptr);
        addBiasColMajor<<<dim3((H1 + 15) / 16, (R + 15) / 16), dim3(16, 16)>>>(d_query_proj.ptr, d_b1.ptr, H1, R);

        // docs_proj [H1 x N] = W1_doc @ docs^T (bias lives on the query side only)
        gemm(handle, d_w1_doc.ptr, H1, doc_dim_, d_docs, N, d_docs_proj.ptr);

        {
            dim3 block(16, 16);
            dim3 grid((H1 + 15) / 16, (cols + 15) / 16);
            addBroadcastQueryDoc<<<grid, block>>>(d_act_a.ptr, d_query_proj.ptr, d_docs_proj.ptr, H1, R, N);
            reluInPlace<<<(H1 * cols + 255) / 256, 256>>>(d_act_a.ptr, H1 * cols);
        }

        GpuBuf* src = &d_act_a;
        GpuBuf* dst = &d_act_b;
        for (size_t i = 0; i + 1 < hidden_sizes.size(); ++i)
        {
            const int prev_h = hidden_sizes[i];
            const int curr_h = hidden_sizes[i + 1];
            gemm(handle, d_w_hidden[i].ptr, curr_h, prev_h, src->ptr, cols, dst->ptr);
            {
                dim3 block(16, 16);
                dim3 grid((curr_h + 15) / 16, (cols + 15) / 16);
                addBiasColMajor<<<grid, block>>>(dst->ptr, d_b_hidden[i].ptr, curr_h, cols);
                reluInPlace<<<(curr_h * cols + 255) / 256, 256>>>(dst->ptr, curr_h * cols);
            }
            std::swap(src, dst);
        }

        const int H_last = hidden_sizes.back();
        gemm(handle, d_w_out.ptr, num_heads_, H_last, src->ptr, cols, d_scores);
        {
            dim3 block(16, 16);
            dim3 grid((num_heads_ + 15) / 16, (cols + 15) / 16);
            addBiasColMajor<<<grid, block>>>(d_scores, d_b_out.ptr, num_heads_, cols);
            sigmoidInPlace<<<(num_heads_ * cols + 255) / 256, 256>>>(d_scores, num_heads_ * cols);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    std::vector<float> infer(const Input& in) override
    {
        GpuBuf d_query(in.query.size());
        d_query.upload(in.query);
        GpuBuf d_docs(in.docs.size());
        d_docs.upload(in.docs);
        GpuBuf d_scores(in.num_reqs * in.num_heads * in.num_docs);

        infer_device(d_query.ptr,
                     d_docs.ptr,
                     d_scores.ptr,
                     in.num_reqs,
                     in.query_dim,
                     in.doc_dim,
                     in.num_docs,
                     in.num_heads);

        // d_scores is col-major [num_heads x (num_reqs * num_docs)], column c = r*N + n
        // → reorder to row-major [num_reqs x num_docs x num_heads]
        auto               raw = d_scores.download();
        std::vector<float> scores(in.num_reqs * in.num_docs * in.num_heads);
        for (int r = 0; r < in.num_reqs; ++r)
            for (int d = 0; d < in.num_docs; ++d)
            {
                int c = r * in.num_docs + d;
                for (int h = 0; h < in.num_heads; ++h)
                    scores[(r * in.num_docs + d) * in.num_heads + h] = raw[h + c * in.num_heads];
            }
        return scores;
    }
};

std::unique_ptr<InferBackend> make_cuda(const Paths& paths, const Input& shape_hint)
{
    return std::make_unique<CudaBackend>(paths, shape_hint);
}
