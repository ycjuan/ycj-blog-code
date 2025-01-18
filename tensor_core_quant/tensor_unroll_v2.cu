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

   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag0;
   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag1;
   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag2;
   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag3;
   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag4;
   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag5;
   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag6;
   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag7;
   wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag0;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag00;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag10;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag20;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag30;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag40;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag50;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag60;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag70;
   wmma::fill_fragment(c_frag00, 0);
   wmma::fill_fragment(c_frag10, 0);
   wmma::fill_fragment(c_frag20, 0);
   wmma::fill_fragment(c_frag30, 0);
   wmma::fill_fragment(c_frag40, 0);
   wmma::fill_fragment(c_frag50, 0);
   wmma::fill_fragment(c_frag60, 0);
   wmma::fill_fragment(c_frag70, 0);

   size_t lda32 = lda / 32;
   size_t ldb32 = ldb / 32;
   for (int i = 0; i < K; i += WMMA_K) {
      int i32 = i / 32;
      size_t aRow0 = warpM * WMMA_M * 8;
      size_t aCol0 = i32;

      size_t aRow1 = aRow0 + WMMA_M;
      size_t aCol1 = i32;

      size_t aRow2 = aRow0 + WMMA_M * 2;
      size_t aCol2 = i32;

      size_t aRow3 = aRow0 + WMMA_M * 3;
      size_t aCol3 = i32;

      size_t aRow4 = aRow0 + WMMA_M * 4;
      size_t aCol4 = i32;

      size_t aRow5 = aRow0 + WMMA_M * 5;
      size_t aCol5 = i32;

      size_t aRow6 = aRow0 + WMMA_M * 6;
      size_t aCol6 = i32;

      size_t aRow7 = aRow0 + WMMA_M * 7;
      size_t aCol7 = i32;

      size_t bRow0 = i32;
      size_t bCol0 = warpN * WMMA_N * 1;

      wmma::load_matrix_sync(a_frag0, a + aRow0 * lda32 + aCol0, lda);
      wmma::load_matrix_sync(a_frag1, a + aRow1 * lda32 + aCol1, lda);
      wmma::load_matrix_sync(a_frag2, a + aRow2 * lda32 + aCol2, lda);
      wmma::load_matrix_sync(a_frag3, a + aRow3 * lda32 + aCol3, lda);
      wmma::load_matrix_sync(a_frag4, a + aRow4 * lda32 + aCol4, lda);
      wmma::load_matrix_sync(a_frag5, a + aRow5 * lda32 + aCol5, lda);
      wmma::load_matrix_sync(a_frag6, a + aRow6 * lda32 + aCol6, lda);
      wmma::load_matrix_sync(a_frag7, a + aRow7 * lda32 + aCol7, lda);
      wmma::load_matrix_sync(b_frag0, b + bCol0 * ldb32 + bRow0, ldb);

      // Perform the matrix multiplication
      wmma::bmma_sync(c_frag00, a_frag0, b_frag0, c_frag00);
      wmma::bmma_sync(c_frag10, a_frag1, b_frag0, c_frag10);
      wmma::bmma_sync(c_frag20, a_frag2, b_frag0, c_frag20);
      wmma::bmma_sync(c_frag30, a_frag3, b_frag0, c_frag30);
      wmma::bmma_sync(c_frag40, a_frag4, b_frag0, c_frag40);
      wmma::bmma_sync(c_frag50, a_frag5, b_frag0, c_frag50);
      wmma::bmma_sync(c_frag60, a_frag6, b_frag0, c_frag60);
      wmma::bmma_sync(c_frag70, a_frag7, b_frag0, c_frag70);
   }

   int cRow0 = warpM * WMMA_M * 8;
   int cRow1 = cRow0 + WMMA_M;
   int cRow2 = cRow0 + WMMA_M * 2;
   int cRow3 = cRow0 + WMMA_M * 3;
   int cRow4 = cRow0 + WMMA_M * 4;
   int cRow5 = cRow0 + WMMA_M * 5;
   int cRow6 = cRow0 + WMMA_M * 6;
   int cRow7 = cRow0 + WMMA_M * 7;

   int cCol0 = warpN * WMMA_N * 1;

#pragma unroll
   for (int i = 0; i < c_frag00.num_elements; i++)
   {
      c_frag00.x[i] = K - c_frag00.x[i];
      c_frag10.x[i] = K - c_frag10.x[i];
      c_frag20.x[i] = K - c_frag20.x[i];
      c_frag30.x[i] = K - c_frag30.x[i];
      c_frag40.x[i] = K - c_frag40.x[i];
      c_frag50.x[i] = K - c_frag50.x[i];
      c_frag60.x[i] = K - c_frag60.x[i];
      c_frag70.x[i] = K - c_frag70.x[i];
   }

   wmma::store_matrix_sync(c + cRow0 * ldc + cCol0, c_frag00, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow1 * ldc + cCol0, c_frag10, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow2 * ldc + cCol0, c_frag20, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow3 * ldc + cCol0, c_frag30, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow4 * ldc + cCol0, c_frag40, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow5 * ldc + cCol0, c_frag50, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow6 * ldc + cCol0, c_frag60, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow7 * ldc + cCol0, c_frag70, ldc, wmma::mem_row_major);
   
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
 
   blockDim.x = 128;
   blockDim.y = 4;

   int unitX = WMMA_M * blockDim.x / 32 * 8;
   int unitY = WMMA_N * blockDim.y * 1;

   gridDim.x = (MATRIX_M + unitX - 1) / unitX;
   gridDim.y = (MATRIX_N + unitY - 1) / unitY;

   printf("\nRunning with wmma (unroll v2)...\n");
   printf("M = %d, N = %d, K = %d.\n", MATRIX_M, MATRIX_N, MATRIX_K);
   cout << "blockDim: " << blockDim.x << " " << blockDim.y << endl;
   cout << "gridDim: " << gridDim.x << " " << gridDim.y << endl;
   CudaTimer timer;
   for (int t = -3; t < setting.kNumTrials; t++)
   {
      if (t == 0)
         timer.tic();
      quantWmmaUnrollKernelV2 <<< gridDim, blockDim >>> (a, b, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K * 32);
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(cudaGetLastError());
   }
   cout << "wmma (unroll v2) took " << timer.tocMs() / setting.kNumTrials << "ms" << endl;
}


