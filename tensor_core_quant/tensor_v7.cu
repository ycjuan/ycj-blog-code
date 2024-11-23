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

__global__ void wmma_example(const unsigned *A, const unsigned *B, int *C, const unsigned m, const unsigned n, const unsigned k)
{
   using namespace nvcuda::wmma::experimental;
   int bx = blockIdx.x * blockDim.y + threadIdx.y;
   int by = blockIdx.y;

   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
   wmma::fill_fragment(c_frag, 0);

   for (int j = 0; j < (k / 128); j++)
   {
      load_matrix_sync(a_frag, A + bx * 8 * k / 32 + j * 128 * 8 / 32, 128);
      load_matrix_sync(b_frag, B + by * 8 * k / 32 + j * 128 * 8 / 32, 128);

      bmma_sync(c_frag, a_frag, b_frag, c_frag, bmmaBitOpXOR, bmmaAccumulateOpPOPC);
   }

#pragma unroll
   for (int i = 0; i < c_frag.num_elements; i++)
      c_frag.x[i] = k - c_frag.x[i];

   store_matrix_sync(C + (bx * 8 * n + by * 8), c_frag, n, wmma::mem_row_major);
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
   dim3 blockDim(32, 2);
   dim3 gridDim(MATRIX_M/16, MATRIX_N/8);

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


