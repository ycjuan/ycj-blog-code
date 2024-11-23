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

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}


#include <mma.h>
using namespace nvcuda;

// The only dimensions currently supported by WMMA
const int WMMA_M = 8;
const int WMMA_N = 8;
const int WMMA_K = 128;


// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
__global__ void wmma_example(const unsigned *A, const unsigned *B, int *C, const unsigned m, const unsigned n, const unsigned k)
{
   using namespace nvcuda::wmma::experimental;
   int bx = blockIdx.x * blockDim.y + threadIdx.y;
   int by = blockIdx.y;
   printf("bx: %d, by: %d, blockIdx.x: %d, blockDim.y: %d, threadIdx.y: %d\n", bx, by, blockIdx.x, blockDim.y, threadIdx.y);
   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
   wmma::fill_fragment(c_frag, 0);

   /*
   printf(" \
      a_frag.num_storage_elements: %d, a_frag.num_elements: %d, \
      b_frag.num_storage_elements: %d, b_frag.num_elements: %d, \
      c_frag.num_storage_elements: %d, c_frag.num_elements: %d\n",
      a_frag.num_storage_elements, a_frag.num_elements,
      b_frag.num_storage_elements, b_frag.num_elements,
      c_frag.num_storage_elements, c_frag.num_elements);
   */

   for (int j = 0; j < (k / 128); j++)
   {
      load_matrix_sync(a_frag, A + bx * 8 * k / 32 + j * 128 * 8 / 32, 128);
      load_matrix_sync(b_frag, B + by * 8 * k / 32 + j * 128 * 8 / 32, 128);

      bmma_sync(c_frag, a_frag, b_frag, c_frag);
   }
   store_matrix_sync(C + (bx * 8 * n + by * 8), c_frag, n, wmma::mem_row_major);
      if (bx == 0 && by == 0 && threadIdx.x == 0)
      {
         printf("A[0] = %u\n", A[0]);
         printf("B[0] = %u\n", B[0]);
         printf("a_frag.x[0]: %u, a_frag.x[1]: %u, a_frag.x[2]: %u, a_frag.x[3]: %u\n", a_frag.x[0], a_frag.x[1], a_frag.x[2], a_frag.x[3]);
         printf("b_frag.x[0]: %u, b_frag.x[1]: %u, b_frag.x[2]: %u, b_frag.x[3]: %u\n", b_frag.x[0], b_frag.x[1], b_frag.x[2], b_frag.x[3]);
         printf("c_frag.x[0]: %d, c_frag.x[1]: %d, c_frag.x[2]: %d, c_frag.x[3]: %d\n", c_frag.x[0], c_frag.x[1], c_frag.x[2], c_frag.x[3]);
         printf("C[0]: %d\n", C[0]);
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
   dim3 blockDim(32, 2);
   dim3 gridDim(MATRIX_M/16, MATRIX_N/8);

   cout << "blockDim: " << blockDim.x << " " << blockDim.y << endl;
   cout << "gridDim: " << gridDim.x << " " << gridDim.y << endl;

   cout << "A[0]: " << a_fp16[0] << endl;
 
   printf("Running with wmma...\n");
   CudaTimer timer;
   for (int t = -3; t < -2; t++)
   {
      if (t == 0)
         timer.tic();
      wmma_example <<< gridDim, blockDim >>> (a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K * 32);
      cudaErrCheck(cudaDeviceSynchronize());
      cudaErrCheck(cudaGetLastError());
   }
   cout << "wmma took " << timer.tocMs() / setting.kNumTrials << "ms" << endl;
}


