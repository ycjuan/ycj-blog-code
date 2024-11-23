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

https://github.com/rgb000000/yolo2_light/blob/6bb99873873f0fde08e8ecde754d5cb0e0ff97b8/src/gpu.cu#L1892
*/

#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>

#include "common.cuh"

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
const int WMMA_K32 = (WMMA_K/32);
const int WMMA_Nx2 = (WMMA_N);


// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
__global__ void wmma_example(T1 *A, T1 *B, T2 *C, int M, int N, int K) {

   int lda = K;
   int ldb = K;
   int ldc = N;
   // total 57%
   int index = blockIdx.x * blockDim.x + threadIdx.x;

   const int lane_id = threadIdx.x % 32;
   const int warp_id = threadIdx.x / 32;
   const int global_warp_id = index / 32;

   const int N_aligned = N + WMMA_Nx2 - (N % WMMA_Nx2);

   int i, j, k, h;
   // 47% = 29 + 10 + 8
   j = global_warp_id % (N_aligned / WMMA_Nx2);
   j = j * WMMA_Nx2;
   { // out_h*out_w - one channel output size [169 - 173056]
      i = global_warp_id / (N_aligned / WMMA_Nx2);
      i = i * WMMA_M;

      int count = 0;
      k = 0;

      if (i < M) // if (i < M)  // l.n - filters [16 - 55 - 1024]
      {
         if (j + WMMA_Nx2 > N)
            j = N - WMMA_Nx2; // must be: j+7 < N
         if (i + WMMA_M > M)
            i = M - WMMA_M; // must be: i+7 < M
                            // Tensor Cores
         using namespace nvcuda;

         wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::experimental::precision::b1, wmma::row_major> a_frag;
         wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::experimental::precision::b1, wmma::col_major> b_frag;
         wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c1_frag;
         wmma::fill_fragment(c1_frag, 0); // !!!! XOR isn't XNOR !!!!!!!!!!

         // 8 x 8 x 4 (uint32_t, 4 * 32 = 128 bit)
         for (; k < K; k += 128) // l.size*l.size*l.c - one filter size [27 - 144 - 9216]
         {
            int64_t A_cur_index = (i * lda + k) / 32;        // index in bits
            int64_t B1_cur_index = (j * ldb + k) / 32;       // index in bits

            // try to use A that is cached in shared memory - poor performance
            // if (i == start_i) wmma::load_matrix_sync(a_frag, &A_s[k / 32], (512 * 32));   // lda = (128*32) bits
            // else wmma::load_matrix_sync(a_frag, (uint32_t *)(A + A_cur_index), lda);   // lda = M

            // lda, ldb - are in bits
            wmma::load_matrix_sync(a_frag, (uint32_t *)(A + A_cur_index), lda); // lda = M

            wmma::load_matrix_sync(b_frag, (uint32_t *)(B + B1_cur_index), ldb); // ldb = K
            wmma::bmma_sync(c1_frag, a_frag, b_frag, c1_frag);                   // XOR-GEMM
         }
         // C[i*ldc + j]
         wmma::store_matrix_sync(&C[i*ldc+j], c1_frag, WMMA_N, wmma::mem_row_major);

      }
   }
}

void quantWMMA(Data data, Setting setting) {

   int MATRIX_M = data.numDocs;
   int MATRIX_N = data.numReqs;
   int MATRIX_K = data.numT1;

   T1 *a_fp16 = data.d_doc;
   T1 *b_fp16 = data.d_req;

   T2 *c_wmma = data.d_rst_wmma;

   cudaEvent_t startWMMA;
   cudaEvent_t stopWMMA;
   
   cudaErrCheck(cudaEventCreate(&startWMMA));
   cudaErrCheck(cudaEventCreate(&stopWMMA));

   int MATRIX_K_BITS = MATRIX_K * sizeof(T1) * 8;
   printf("\nM = %d, N = %d, K = %d, K_BITS = %d.\n\n", MATRIX_M, MATRIX_N, MATRIX_K, MATRIX_K_BITS);
   
   // First: using WMMA

   int WARP_SIZE = 32;
   int blockSize = 256;
   int size = (MATRIX_M / 8) * (MATRIX_N / 8);
   int gridSize = (size + blockSize - 1) / blockSize;
   cout << "gridSize (WMMA): " << gridSize << endl;

   printf("Running with wmma...\n");
   cudaErrCheck(cudaEventRecord(startWMMA));
   wmma_example <<< gridSize, blockSize >>> (a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K_BITS);
   cudaErrCheck(cudaEventRecord(stopWMMA));
   cudaErrCheck(cudaEventSynchronize(stopWMMA));

   float wmmaTime;
   cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
   printf("wmma took %fms\n", wmmaTime);

   cudaErrCheck(cudaEventDestroy(startWMMA));
   cudaErrCheck(cudaEventDestroy(stopWMMA));
}


