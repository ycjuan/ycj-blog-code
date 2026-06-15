#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------
#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t err = (call);                                                                                      \
        if (err != cudaSuccess)                                                                                        \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));                           \
    } while (0)

#define CHECK_CUBLAS(call)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        cublasStatus_t st = (call);                                                                                    \
        if (st != CUBLAS_STATUS_SUCCESS)                                                                               \
            throw std::runtime_error("cuBLAS error: " + std::to_string(static_cast<int>(st)));                         \
    } while (0)

// ---------------------------------------------------------------------------
// Load raw float32 binary file into a host vector
// ---------------------------------------------------------------------------
std::vector<float> loadBin(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("Cannot open: " + path);
    size_t bytes = f.tellg();
    f.seekg(0);
    std::vector<float> data(bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

// ---------------------------------------------------------------------------
// RAII GPU buffer
// ---------------------------------------------------------------------------
struct GpuBuf
{
    float* ptr = nullptr;
    size_t n   = 0;

    GpuBuf() = default;
    GpuBuf(size_t n_)
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

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Adds bias b[row] to every element of a col-major matrix A[rows x cols].
// Each column is one doc; each row is one hidden unit.
__global__ void addBiasColMajor(float* A, const float* b, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols)
        A[col * rows + row] += b[row];
}

// ReLU in-place over n elements.
__global__ void reluInPlace(float* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = fmaxf(0.0f, x[i]);
}

// Sigmoid in-place over n elements.
__global__ void sigmoidInPlace(float* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = 1.0f / (1.0f + expf(-x[i]));
}

// ---------------------------------------------------------------------------
// cuBLAS helper: C [out_dim x N, col-major] = W [out_dim x in_dim, row-major]
//                                             @ X [in_dim  x N,  col-major]
//
// W is stored row-major (PyTorch default): W[i,j] = W_ptr[i*in_dim + j].
// Viewed as col-major that is W^T [in_dim x out_dim] with lda = in_dim.
// Using CUBLAS_OP_T on W^T recovers W, giving C = W @ X.
// ---------------------------------------------------------------------------
void gemm(cublasHandle_t handle, const float* W, int out_dim, int in_dim, const float* X, int N, float* C)
{
    const float alpha = 1.0f, beta = 0.0f;
    // C [out_dim x N] = op(W^T) [out_dim x in_dim] @ X [in_dim x N]
    // In cuBLAS col-major: m=out_dim, n=N, k=in_dim
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             out_dim,
                             N,
                             in_dim,
                             &alpha,
                             W,
                             in_dim, // A = W^T col-major, lda = in_dim
                             X,
                             in_dim, // B = X col-major,   ldb = in_dim
                             &beta,
                             C,
                             out_dim)); // C col-major,       ldc = out_dim
}

// ---------------------------------------------------------------------------
// cuBLAS helper: y [out_dim, col-major] = W [out_dim x in_dim, row-major] @ x [in_dim]
// ---------------------------------------------------------------------------
void gemv(cublasHandle_t handle, const float* W, int out_dim, int in_dim, const float* x, float* y)
{
    const float alpha = 1.0f, beta = 0.0f;
    // W row-major [out_dim x in_dim] = W^T col-major [in_dim x out_dim], lda=in_dim
    // CUBLAS_OP_T on that gives W, result is y [out_dim]
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, in_dim, out_dim, &alpha, W, in_dim, x, 1, &beta, y, 1));
}

// ---------------------------------------------------------------------------
// launch helpers
// ---------------------------------------------------------------------------
static void launchAddBias(float* A, const float* b, int rows, int cols)
{
    dim3 block(16, 16);
    dim3 grid((rows + 15) / 16, (cols + 15) / 16);
    addBiasColMajor<<<grid, block>>>(A, b, rows, cols);
}

static void launchRelu(float* x, int n) { reluInPlace<<<(n + 255) / 256, 256>>>(x, n); }

static void launchSigmoid(float* x, int n) { sigmoidInPlace<<<(n + 255) / 256, 256>>>(x, n); }

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    // --- Model config (must match export.py) ---
    const int              query_dim    = 64;
    const int              doc_dim      = 128;
    const std::vector<int> hidden_sizes = { 256, 128 };
    const int              num_heads    = 2;
    const int              num_docs     = 5;
    const std::string      wdir         = "../weights/";

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // --- Upload weights ---
    auto upload = [&](const std::string& name)
    {
        GpuBuf buf;
        auto   h = loadBin(wdir + name);
        buf      = GpuBuf(h.size());
        buf.upload(h);
        return buf;
    };

    GpuBuf d_w1_query = upload("w1_query.bin"); // [H1 x Dq] row-major
    GpuBuf d_b1       = upload("b1.bin");       // [H1]
    GpuBuf d_w1_doc   = upload("w1_doc.bin");   // [H1 x Dd] row-major

    std::vector<GpuBuf> d_w_hidden(hidden_sizes.size() - 1);
    std::vector<GpuBuf> d_b_hidden(hidden_sizes.size() - 1);
    for (size_t i = 0; i + 1 < hidden_sizes.size(); ++i)
    {
        d_w_hidden[i] = upload("w_hidden_" + std::to_string(i) + ".bin");
        d_b_hidden[i] = upload("b_hidden_" + std::to_string(i) + ".bin");
    }
    GpuBuf d_w_out = upload("w_out.bin"); // [num_heads x H_last] row-major
    GpuBuf d_b_out = upload("b_out.bin"); // [num_heads]

    // --- Upload inputs (random, same shape as export.py dummy inputs) ---
    std::vector<float> h_query(query_dim, 0.5f);
    std::vector<float> h_docs(num_docs * doc_dim, 0.5f);

    GpuBuf d_query(query_dim);
    d_query.upload(h_query);

    // docs row-major [num_docs x doc_dim] == col-major [doc_dim x num_docs]: same memory layout
    GpuBuf d_docs(num_docs * doc_dim);
    d_docs.upload(h_docs);

    // --- Activation buffers (col-major: [dim x num_docs]) ---
    const int H1   = hidden_sizes[0];
    const int maxH = *std::max_element(hidden_sizes.begin(), hidden_sizes.end());
    GpuBuf    d_act_a(maxH * num_docs); // ping
    GpuBuf    d_act_b(maxH * num_docs); // pong
    GpuBuf    d_query_proj(H1);         // [H1] — query contribution broadcast as bias
    GpuBuf    d_scores(num_heads * num_docs);

    // --- Forward pass ---

    // Layer 1a: query_proj [H1] = W1_query [H1 x Dq] @ query [Dq] + b1
    gemv(handle, d_w1_query.ptr, H1, query_dim, d_query.ptr, d_query_proj.ptr);
    { // add b1 to query_proj (both [H1])
        const float alpha = 1.0f;
        CHECK_CUBLAS(cublasSaxpy(handle, H1, &alpha, d_b1.ptr, 1, d_query_proj.ptr, 1));
    }

    // Layer 1b: act_a [H1 x N] = W1_doc [H1 x Dd] @ docs [Dd x N]
    //           + broadcast query_proj as bias, then ReLU
    gemm(handle, d_w1_doc.ptr, H1, doc_dim, d_docs.ptr, num_docs, d_act_a.ptr);
    launchAddBias(d_act_a.ptr, d_query_proj.ptr, H1, num_docs);
    launchRelu(d_act_a.ptr, H1 * num_docs);

    // Hidden layers: ping-pong between d_act_a and d_act_b
    GpuBuf* src = &d_act_a;
    GpuBuf* dst = &d_act_b;
    for (size_t i = 0; i + 1 < hidden_sizes.size(); ++i)
    {
        const int prev_h = hidden_sizes[i];
        const int curr_h = hidden_sizes[i + 1];
        gemm(handle, d_w_hidden[i].ptr, curr_h, prev_h, src->ptr, num_docs, dst->ptr);
        launchAddBias(dst->ptr, d_b_hidden[i].ptr, curr_h, num_docs);
        launchRelu(dst->ptr, curr_h * num_docs);
        std::swap(src, dst);
    }

    // Output layer: scores [num_heads x N] = W_out [num_heads x H_last] @ last_act [H_last x N] + b_out
    const int H_last = hidden_sizes.back();
    gemm(handle, d_w_out.ptr, num_heads, H_last, src->ptr, num_docs, d_scores.ptr);
    launchAddBias(d_scores.ptr, d_b_out.ptr, num_heads, num_docs);
    launchSigmoid(d_scores.ptr, num_heads * num_docs);

    CHECK_CUDA(cudaDeviceSynchronize());

    // --- Print results ---
    // d_scores is [num_heads x num_docs] col-major → read as [num_docs x num_heads]
    auto h_scores = d_scores.download();
    std::cout << "Scores shape: [" << num_docs << ", " << num_heads << "]\n";
    std::cout << "Scores:\n";
    for (int d = 0; d < num_docs; ++d)
    {
        for (int h = 0; h < num_heads; ++h)
            std::cout << h_scores[h + d * num_heads] << " ";
        std::cout << "\n";
    }

    cublasDestroy(handle);
    return 0;
}
