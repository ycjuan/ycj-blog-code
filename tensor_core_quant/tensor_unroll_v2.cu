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

__global__ void quantWmmaUnrollKernelV2(const unsigned *a, const unsigned *b, int *c, const unsigned M, const unsigned N, const unsigned K)
{
   using namespace nvcuda::wmma::experimental;
   size_t lda = K;
   size_t ldb = K;
   size_t ldc = N;

   // Tile using a 2D grid
   size_t warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   size_t warpN = (blockIdx.y * blockDim.y + threadIdx.y);

   // 8: WWMA_M, 8: WMMA_N, 128: WMMA_K
   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag0;
   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag1;
   wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag0;
   wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag1;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag00;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag01;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag10;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag11;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag20;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag21;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag30;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag31;
   wmma::fill_fragment(c_frag00, 0);
   wmma::fill_fragment(c_frag01, 0);
   wmma::fill_fragment(c_frag10, 0);
   wmma::fill_fragment(c_frag11, 0);
   wmma::fill_fragment(c_frag20, 0);
   wmma::fill_fragment(c_frag21, 0);
   wmma::fill_fragment(c_frag30, 0);
   wmma::fill_fragment(c_frag31, 0);

   size_t lda32 = lda / 32; // 32: number of bytes of int32
   size_t ldb32 = ldb / 32; // 32: number of bytes of int32
   for (int i = 0; i < K; i += WMMA_K) {
      int i32 = i / 32; // 32: number of bytes of int32
      size_t aRow0 = warpM * WMMA_M * 2; // 4: because we do 4x unrolling
      size_t aCol0 = i32;

      size_t aRow1 = aRow0 + WMMA_M;
      size_t aCol1 = i32;

      size_t bRow0 = i32;
      size_t bCol0 = warpN * WMMA_N * 2; // 4: because we do 4x unrolling

      size_t bRow1 = i32;
      size_t bCol1 = bCol0 + WMMA_N;

      wmma::load_matrix_sync(a_frag0, a + aRow0 * lda32 + aCol0, lda);
      wmma::load_matrix_sync(a_frag1, a + aRow1 * lda32 + aCol1, lda);
      wmma::load_matrix_sync(b_frag0, b + bCol0 * ldb32 + bRow0, ldb);
      wmma::load_matrix_sync(b_frag1, b + bCol1 * ldb32 + bRow1, ldb);

      // Perform the matrix multiplication
      wmma::bmma_sync(c_frag00, a_frag0, b_frag0, c_frag00);
      wmma::bmma_sync(c_frag01, a_frag0, b_frag1, c_frag01);
      wmma::bmma_sync(c_frag10, a_frag1, b_frag0, c_frag10);
      wmma::bmma_sync(c_frag11, a_frag1, b_frag1, c_frag11);

   }

   int cRow0 = warpM * WMMA_M * 2; // 4: because we do 4x unrolling
   int cRow1 = cRow0 + WMMA_M;

   int cCol0 = warpN * WMMA_N * 2; // 4: because we do 4x unrolling
   int cCol1 = cCol0 + WMMA_N;

#pragma unroll
   for (int i = 0; i < c_frag00.num_elements; i++)
   {
      c_frag00.x[i] = K - c_frag00.x[i];
      c_frag01.x[i] = K - c_frag01.x[i];
      c_frag10.x[i] = K - c_frag10.x[i];
      c_frag11.x[i] = K - c_frag11.x[i];
      c_frag20.x[i] = K - c_frag20.x[i];
      c_frag21.x[i] = K - c_frag21.x[i];
      c_frag30.x[i] = K - c_frag30.x[i];
      c_frag31.x[i] = K - c_frag31.x[i];
   }

   wmma::store_matrix_sync(c + cRow0 * ldc + cCol0, c_frag00, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow0 * ldc + cCol1, c_frag01, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow1 * ldc + cCol0, c_frag10, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow1 * ldc + cCol1, c_frag11, ldc, wmma::mem_row_major);
   
}

void quantWmmaUnrollV2(Data data, Setting setting) {

   int MATRIX_M = data.numDocs;
   int MATRIX_N = data.numReqs;
   int MATRIX_K = data.numInt;

   T_QUANT *a = data.d_doc;
   T_QUANT *b = data.d_req;

   T_RST *c_wmma = data.d_rstWmmaUnroll;

   
   dim3 gridDim;
   dim3 blockDim;
 
   blockDim.x = 512;
   blockDim.y = 1;

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32) / 2; // 32: warpSize; 4: because we do 4x unrolling
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y) / 2; // 4: because we do 4x unrolling

   printf("\nRunning with wmma (unroll)...\n");
   printf("M = %d, N = %d, K = %d.\n", MATRIX_M, MATRIX_N, MATRIX_K);
   cout << "blockDim: " << blockDim.x << " " << blockDim.y << endl;
   cout << "gridDim: " << gridDim.x << " " << gridDim.y << endl;
   CudaTimer timer;
   for (int t = -3; t < setting.kNumTrials; t++)
   {
      if (t == 0)
         timer.tic();
      quantWmmaUnrollKernelV2 <<< gridDim, blockDim >>> (a, b, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K * 32); // 32: number of bytes of int32
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(cudaGetLastError());
   }
   cout << "wmma (unroll) took " << timer.tocMs() / setting.kNumTrials << "ms" << endl;
}


