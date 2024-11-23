/* Copyright (c) 1993-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
This file is modified from: 

https://github.com/NVIDIA-developer-blog/code-samples/blob/708ce9137eb5ac7682f788e5d5b8279c7e2578ed/posts/tensor-cores/simpleTensorCoreGEMM.cu

https://github.com/pnnl/TCBNN/blob/de4713445fd1cd772ad176080a0ff61a5f862e3b/bmm/tensorcore_kernel.cu#L336
*/

#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>

#include "common.cuh"
#include "util.cuh"

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#include <mma.h>
using namespace nvcuda;

// The only dimensions currently supported by WMMA
const int WMMA_M = 8;
const int WMMA_N = 8;
const int WMMA_K = 128;

__global__ void wmma_example(const unsigned *a, const unsigned *b, int *c, const unsigned M, const unsigned N, const unsigned K)
{
   using namespace nvcuda::wmma::experimental;
   int lda = K;
   int ldb = K;
   int ldc = M;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
   wmma::fill_fragment(c_frag, 0);

   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
         wmma::load_matrix_sync(a_frag, a + aRow * lda / 32 + aCol, lda);
         wmma::load_matrix_sync(b_frag, b + bCol * ldb / 32 + bRow, ldb);

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
      wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
   }
}

void quantWMMA(Data data, Setting setting) {

   int MATRIX_M = data.numDocs;
   int MATRIX_N = data.numReqs;
   int MATRIX_K = data.numT1;

   T1 *a_fp16 = data.d_doc;
   T1 *b_fp16 = data.d_req;

   T2 *c_wmma = data.d_rst_wmma;

   printf("\nM = %d, N = %d, K = %d.\n\n", MATRIX_M, MATRIX_N, MATRIX_K);
   
   // First: using WMMA
   dim3 gridDim;
   dim3 blockDim;
 
   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = 128;
   blockDim.y = 4;

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

   cout << "blockDim: " << blockDim.x << " " << blockDim.y << endl;
   cout << "gridDim: " << gridDim.x << " " << gridDim.y << endl;

   printf("Running with wmma...\n");
   CudaTimer timer;
   for (int t = -3; t < setting.kNumTrials; t++)
   {
      if (t == 0)
         timer.tic();
      wmma_example <<< gridDim, blockDim >>> (a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K * 32);
      cudaErrCheck(cudaDeviceSynchronize());
      cudaErrCheck(cudaGetLastError());
   }
   cout << "wmma took " << timer.tocMs() / setting.kNumTrials << "ms" << endl;
}


