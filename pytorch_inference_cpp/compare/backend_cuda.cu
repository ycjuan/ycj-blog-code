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

void gemv(cublasHandle_t h, const float* W, int out_dim, int in_dim, const float* x, float* y)
{
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemv(h, CUBLAS_OP_T, in_dim, out_dim, &alpha, W, in_dim, x, 1, &beta, y, 1));
}

} // namespace

std::vector<float> run_cuda(const Paths& paths, const Input& in)
{
    const int          num_docs = in.num_docs;
    const std::string& wdir     = paths.weights_dir;

    // Infer hidden_sizes from weight files: keep loading w_hidden_i until missing
    std::vector<int> hidden_sizes;
    {
        auto w1 = loadBin(wdir + "w1_query.bin"); // [H1 x query_dim]
        hidden_sizes.push_back(static_cast<int>(w1.size()) / in.query_dim);
        for (int i = 0;; ++i)
        {
            std::ifstream f(wdir + "w_hidden_" + std::to_string(i) + ".bin", std::ios::binary);
            if (!f)
                break;
            f.seekg(0, std::ios::end);
            size_t bytes = f.tellg();
            int    prev  = hidden_sizes.back();
            hidden_sizes.push_back(static_cast<int>(bytes / sizeof(float)) / prev);
        }
    }

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    auto upload = [&](const std::string& name)
    {
        auto   h = loadBin(wdir + name);
        GpuBuf buf(h.size());
        buf.upload(h);
        return buf;
    };

    const int H1 = hidden_sizes[0];

    GpuBuf d_w1_query = upload("w1_query.bin");
    GpuBuf d_b1       = upload("b1.bin");
    GpuBuf d_w1_doc   = upload("w1_doc.bin");

    std::vector<GpuBuf> d_w_hidden(hidden_sizes.size() - 1);
    std::vector<GpuBuf> d_b_hidden(hidden_sizes.size() - 1);
    for (size_t i = 0; i + 1 < hidden_sizes.size(); ++i)
    {
        d_w_hidden[i] = upload("w_hidden_" + std::to_string(i) + ".bin");
        d_b_hidden[i] = upload("b_hidden_" + std::to_string(i) + ".bin");
    }
    GpuBuf d_w_out = upload("w_out.bin");
    GpuBuf d_b_out = upload("b_out.bin");

    // Upload query and docs (docs transposed to col-major)
    GpuBuf d_query(in.query_dim);
    d_query.upload(in.query);

    // docs row-major [num_docs x doc_dim] == col-major [doc_dim x num_docs]: same memory layout
    GpuBuf d_docs(in.docs.size());
    d_docs.upload(in.docs);

    const int maxH = *std::max_element(hidden_sizes.begin(), hidden_sizes.end());
    GpuBuf    d_query_proj(H1);
    GpuBuf    d_act_a(maxH * num_docs);
    GpuBuf    d_act_b(maxH * num_docs);
    GpuBuf    d_scores(in.num_heads * num_docs);

    // Layer 1: query_proj = W1_query @ query + b1
    gemv(handle, d_w1_query.ptr, H1, in.query_dim, d_query.ptr, d_query_proj.ptr);
    {
        const float alpha = 1.0f;
        CHECK_CUBLAS(cublasSaxpy(handle, H1, &alpha, d_b1.ptr, 1, d_query_proj.ptr, 1));
    }

    // Layer 1: act = ReLU(W1_doc @ docs + query_proj)
    gemm(handle, d_w1_doc.ptr, H1, in.doc_dim, d_docs.ptr, num_docs, d_act_a.ptr);
    {
        dim3 block(16, 16);
        dim3 grid((H1 + 15) / 16, (num_docs + 15) / 16);
        addBiasColMajor<<<grid, block>>>(d_act_a.ptr, d_query_proj.ptr, H1, num_docs);
        reluInPlace<<<(H1 * num_docs + 255) / 256, 256>>>(d_act_a.ptr, H1 * num_docs);
    }

    // Hidden layers
    GpuBuf* src = &d_act_a;
    GpuBuf* dst = &d_act_b;
    for (size_t i = 0; i + 1 < hidden_sizes.size(); ++i)
    {
        const int prev_h = hidden_sizes[i];
        const int curr_h = hidden_sizes[i + 1];
        gemm(handle, d_w_hidden[i].ptr, curr_h, prev_h, src->ptr, num_docs, dst->ptr);
        {
            dim3 block(16, 16);
            dim3 grid((curr_h + 15) / 16, (num_docs + 15) / 16);
            addBiasColMajor<<<grid, block>>>(dst->ptr, d_b_hidden[i].ptr, curr_h, num_docs);
            reluInPlace<<<(curr_h * num_docs + 255) / 256, 256>>>(dst->ptr, curr_h * num_docs);
        }
        std::swap(src, dst);
    }

    // Output layer + sigmoid
    const int H_last = hidden_sizes.back();
    gemm(handle, d_w_out.ptr, in.num_heads, H_last, src->ptr, num_docs, d_scores.ptr);
    {
        dim3 block(16, 16);
        dim3 grid((in.num_heads + 15) / 16, (num_docs + 15) / 16);
        addBiasColMajor<<<grid, block>>>(d_scores.ptr, d_b_out.ptr, in.num_heads, num_docs);
        sigmoidInPlace<<<(in.num_heads * num_docs + 255) / 256, 256>>>(d_scores.ptr, in.num_heads * num_docs);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // d_scores is [num_heads x num_docs] col-major → convert to [num_docs x num_heads] row-major
    auto               raw = d_scores.download();
    std::vector<float> scores(in.num_heads * num_docs);
    for (int d = 0; d < num_docs; ++d)
        for (int h = 0; h < in.num_heads; ++h)
            scores[d * in.num_heads + h] = raw[h + d * in.num_heads];

    cublasDestroy(handle);
    return scores;
}
