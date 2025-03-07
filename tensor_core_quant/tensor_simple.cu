/*
This file is modified from: 

https://github.com/NVIDIA-developer-blog/code-samples/blob/708ce9137eb5ac7682f788e5d5b8279c7e2578ed/posts/tensor-cores/simpleTensorCoreGEMM.cu

https://github.com/pnnl/TCBNN/blob/de4713445fd1cd772ad176080a0ff61a5f862e3b/bmm/tensorcore_kernel.cu#L336

Thanks to the authors of the original code!
*/

#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>

#include "common.cuh"
#include "util.cuh"

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

#include <mma.h>
using namespace nvcuda;

const int WMMA_M = 8;
const int WMMA_N = 8;
const int WMMA_K = 128;

__global__ void quantWmmaKernel(const unsigned *a, const unsigned *b, int *c, const unsigned M, const unsigned N, const unsigned K)
{
   using namespace nvcuda::wmma::experimental;
   size_t lda = K;
   size_t ldb = K;
   size_t ldc = N;

   // Tile using a 2D grid
   size_t warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   size_t warpN = (blockIdx.y * blockDim.y + threadIdx.y);

   // 8: WWMA_M, 8: WMMA_N, 128: WMMA_K
   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
   wmma::fill_fragment(c_frag, 0);

   for (int i = 0; i < K; i += WMMA_K) {
      size_t aRow = warpM * WMMA_M;
      size_t aCol = i / 32; // 32: number of bytes of int32

      size_t bRow = i / 32; // 32: number of bytes of int32
      size_t bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
         wmma::load_matrix_sync(a_frag, a + aRow * lda / 32 + aCol, lda); // 32: number of bytes of int32
         wmma::load_matrix_sync(b_frag, b + bCol * ldb / 32 + bRow, ldb); // 32: number of bytes of int32

         // Perform the matrix multiplication
         wmma::bmma_sync(c_frag, a_frag, b_frag, c_frag);

      }
   }

   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

#pragma unroll
   for (int i = 0; i < c_frag.num_elements; i++)
      c_frag.x[i] = K - c_frag.x[i];

   if (cRow < M && cCol < N) {
      wmma::store_matrix_sync(c + cRow * ldc + cCol, c_frag, ldc, wmma::mem_row_major);
   }
}

void quantWmmaSimple(Data data, Setting setting) {

   int MATRIX_M = data.numDocs;
   int MATRIX_N = data.numReqs;
   int MATRIX_K = data.numInt;

   T_QUANT *a = data.d_doc;
   T_QUANT *b = data.d_req;

   T_RST *c_wmma = data.d_rstWmmaSimple;

   
   // First: using WMMA
   dim3 gridDim;
   dim3 blockDim;
 
   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = 128;
   blockDim.y = 4;

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32); // 32: warpSize
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

   printf("\nRunning with wmma (simple)...\n");
   printf("M = %d, N = %d, K = %d.\n", MATRIX_M, MATRIX_N, MATRIX_K);
   cout << "blockDim: " << blockDim.x << " " << blockDim.y << endl;
   cout << "gridDim: " << gridDim.x << " " << gridDim.y << endl;
   CudaTimer timer;
   for (int t = -3; t < setting.kNumTrials; t++)
   {
      if (t == 0)
         timer.tic();
      quantWmmaKernel <<< gridDim, blockDim >>> (a, b, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K * 32);
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(cudaGetLastError());
   }
   cout << "wmma (simple) took " << timer.tocMs() / setting.kNumTrials << "ms" << endl;
}


