#include <sstream>

#include "methods.cuh"

using namespace std;

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

__device__ __host__ size_t getMemAddrRowMajor(int i, int j, int numRows, int numCols)
{
    return i * numCols + j;
}

__device__ __host__ size_t getMemAddrColMajor(int i, int j, int numRows, int numCols)
{
    return j * numRows + i;
}

void h2h_r2r(Data &data)
{
    #pragma omp parallel for
    for (size_t i = 0; i < data.numRows; i++)
    {
        for (size_t j = 0; j < data.numCols; j++)
        {
            T src = data.hp_data_src[getMemAddrRowMajor(i, j, data.numRows, data.numCols)];
            T &dst = data.hp_data_dst[getMemAddrRowMajor(i, j, data.numRows, data.numCols)];
            dst = src;
        }
    }
}

void h2h_r2c(Data &data)
{
    #pragma omp parallel for
    for (size_t i = 0; i < data.numRows; i++)
    {
        for (size_t j = 0; j < data.numCols; j++)
        {
            T src = data.hp_data_src[getMemAddrRowMajor(i, j, data.numRows, data.numCols)];
            T &dst = data.hp_data_dst[getMemAddrColMajor(i, j, data.numRows, data.numCols)];
            dst = src;
        }
    }
}

void h2h_c2r(Data &data)
{
    #pragma omp parallel for
    for (size_t i = 0; i < data.numRows; i++)
    {
        for (size_t j = 0; j < data.numCols; j++)
        {
            T src = data.hp_data_src[getMemAddrColMajor(i, j, data.numRows, data.numCols)];
            T &dst = data.hp_data_dst[getMemAddrRowMajor(i, j, data.numRows, data.numCols)];
            dst = src;
        }
    }
}

void h2h_c2c(Data &data)
{
    #pragma omp parallel for
    for (size_t j = 0; j < data.numCols; j++)
    {
        for (size_t i = 0; i < data.numRows; i++)
        {
            T src = data.hp_data_src[getMemAddrColMajor(i, j, data.numRows, data.numCols)];
            T &dst = data.hp_data_dst[getMemAddrColMajor(i, j, data.numRows, data.numCols)];
            dst = src;
        }
    }
}

__global__ void r2r_kernel_row_first(T *src, T *dst, size_t numRows, size_t numCols)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numRows && j < numCols)
    {
        dst[getMemAddrRowMajor(i, j, numRows, numCols)] = src[getMemAddrRowMajor(i, j, numRows, numCols)];
    }
}

__global__ void r2r_kernel_col_first(T *src, T *dst, size_t numRows, size_t numCols)
{
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numRows && j < numCols)
    {
        dst[getMemAddrRowMajor(i, j, numRows, numCols)] = src[getMemAddrRowMajor(i, j, numRows, numCols)];
    }
}

__global__ void r2c_kernel_row_first(T *src, T *dst, size_t numRows, size_t numCols)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numRows && j < numCols)
    {
        dst[getMemAddrColMajor(i, j, numRows, numCols)] = src[getMemAddrRowMajor(i, j, numRows, numCols)];
    }
}

__global__ void r2c_kernel_col_first(T *src, T *dst, size_t numRows, size_t numCols)
{
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numRows && j < numCols)
    {
        dst[getMemAddrColMajor(i, j, numRows, numCols)] = src[getMemAddrRowMajor(i, j, numRows, numCols)];
    }
}

__global__ void c2r_kernel_row_first(T *src, T *dst, size_t numRows, size_t numCols)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numRows && j < numCols)
    {
        dst[getMemAddrRowMajor(i, j, numRows, numCols)] = src[getMemAddrColMajor(i, j, numRows, numCols)];
    }
}

__global__ void c2r_kernel_col_first(T *src, T *dst, size_t numRows, size_t numCols)
{
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numRows && j < numCols)
    {
        dst[getMemAddrRowMajor(i, j, numRows, numCols)] = src[getMemAddrColMajor(i, j, numRows, numCols)];
    }
}

__global__ void c2c_kernel_row_first(T *src, T *dst, size_t numRows, size_t numCols)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numRows && j < numCols)
    {
        dst[getMemAddrColMajor(i, j, numRows, numCols)] = src[getMemAddrColMajor(i, j, numRows, numCols)];
    }
}

__global__ void c2c_kernel_col_first(T *src, T *dst, size_t numRows, size_t numCols)
{
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numRows && j < numCols)
    {
        dst[getMemAddrColMajor(i, j, numRows, numCols)] = src[getMemAddrColMajor(i, j, numRows, numCols)];
    }
}

void h2d_r2r(Data &data)
{
    dim3 block(32, 32);
    dim3 grid((data.numCols + block.x - 1) / block.x, (data.numRows + block.y - 1) / block.y);
    r2r_kernel_col_first<<<grid, block>>>(data.hp_data_src, data.d_data_dst, data.numRows, data.numCols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

void h2d_r2c(Data &data)
{
    dim3 block(32, 32);
    dim3 grid((data.numCols + block.x - 1) / block.x, (data.numRows + block.y - 1) / block.y);
    r2c_kernel_col_first<<<grid, block>>>(data.hp_data_src, data.d_data_dst, data.numRows, data.numCols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

void h2d_c2r(Data &data)
{
    dim3 block(32, 32);
    dim3 grid((data.numRows + block.x - 1) / block.x, (data.numCols + block.y - 1) / block.y);
    c2r_kernel_row_first<<<grid, block>>>(data.hp_data_src, data.d_data_dst, data.numRows, data.numCols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

void h2d_c2c(Data &data)
{
    dim3 block(32, 32);
    dim3 grid((data.numRows + block.x - 1) / block.x, (data.numCols + block.y - 1) / block.y);
    c2c_kernel_row_first<<<grid, block>>>(data.hp_data_src, data.d_data_dst, data.numRows, data.numCols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

void d2h_r2r(Data &data)
{
    dim3 block(32, 32);
    dim3 grid((data.numCols + block.x - 1) / block.x, (data.numRows + block.y - 1) / block.y);
    r2r_kernel_col_first<<<grid, block>>>(data.d_data_src, data.hp_data_dst, data.numRows, data.numCols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

void d2h_r2c(Data &data)
{
    dim3 block(32, 32);
    dim3 grid((data.numRows + block.x - 1) / block.x, (data.numCols + block.y - 1) / block.y);
    r2c_kernel_row_first<<<grid, block>>>(data.d_data_src, data.hp_data_dst, data.numRows, data.numCols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

void d2h_c2r(Data &data)
{
    dim3 block(32, 32);
    dim3 grid((data.numCols + block.x - 1) / block.x, (data.numRows + block.y - 1) / block.y);
    c2r_kernel_col_first<<<grid, block>>>(data.d_data_src, data.hp_data_dst, data.numRows, data.numCols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

void d2h_c2c(Data &data)
{
    dim3 block(32, 32);
    dim3 grid((data.numRows + block.x - 1) / block.x, (data.numCols + block.y - 1) / block.y);
    c2c_kernel_row_first<<<grid, block>>>(data.d_data_src, data.hp_data_dst, data.numRows, data.numCols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

void d2d_r2r(Data &data)
{
    dim3 block(32, 32);
    dim3 grid((data.numCols + block.x - 1) / block.x, (data.numRows + block.y - 1) / block.y);
    r2r_kernel_col_first<<<grid, block>>>(data.d_data_src, data.d_data_dst, data.numRows, data.numCols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

void d2d_r2c(Data &data)
{
    dim3 block(32, 32);
    dim3 grid((data.numRows + block.x - 1) / block.x, (data.numCols + block.y - 1) / block.y);
    r2c_kernel_row_first<<<grid, block>>>(data.d_data_src, data.d_data_dst, data.numRows, data.numCols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

void d2d_c2r(Data &data)
{
    dim3 block(32, 32);
    dim3 grid((data.numCols + block.x - 1) / block.x, (data.numRows + block.y - 1) / block.y);
    c2r_kernel_col_first<<<grid, block>>>(data.d_data_src, data.d_data_dst, data.numRows, data.numCols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

void d2d_c2c(Data &data)
{
    dim3 block(32, 32);
    dim3 grid((data.numRows + block.x - 1) / block.x, (data.numCols + block.y - 1) / block.y);
    c2c_kernel_row_first<<<grid, block>>>(data.d_data_src, data.d_data_dst, data.numRows, data.numCols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}