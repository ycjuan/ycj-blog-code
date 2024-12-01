/*
This file is modified from: 

https://github.com/NVIDIA-developer-blog/code-samples/blob/708ce9137eb5ac7682f788e5d5b8279c7e2578ed/posts/tensor-cores/simpleTensorCoreGEMM.cu

*/

#include <stdio.h>
#include <cublas_v2.h>
#include <mma.h>
#include <set>

#include "common.cuh"
#include "util.cuh"

using namespace nvcuda;

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

struct PairBlock
{
   int docIdx;
   int reqIdx;
   int docBlockIdx;
   int docBlockOffset;
   int reqBlockIdx;
   int reqBlockOffset;
   float score;
};

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

void convertToPairBlock(Data data, PairBlock *pairBlock)
{
   for (int i = 0; i < data.numPairsToScore; i++)
   {
      Pair pair = data.d_PairsToScore[i];
      pairBlock[i].docIdx = pair.docIdx;
      pairBlock[i].reqIdx = pair.reqIdx;
      pairBlock[i].docBlockIdx = pair.docIdx / WMMA_M;
      pairBlock[i].docBlockOffset = pair.docIdx % WMMA_M;
      pairBlock[i].reqBlockIdx = pair.reqIdx / WMMA_N;
      pairBlock[i].reqBlockOffset = pair.reqIdx % WMMA_N;
   }
}

void dedupPairBlock(PairBlock *pairBlock, int numPairsToScore, int numPairBlocks)
{
   numPairBlocks = 0;
   PairBlock currPair = pairBlock[0];
   for (int i = 1; i < numPairsToScore; i++)
   {
      PairBlock pair = pairBlock[i];
      if (pair.docBlockIdx == currPair.docBlockIdx && pair.reqBlockIdx == currPair.reqBlockIdx)
      {
         continue;
      }
      else
      {
         pairBlock[numPairBlocks++] = currPair;
         currPair = pair;
      }
   }
}

__global__ void tensorSimpleKernel(T *a, T *b, float *c, int M, int N, int K) {
   // Leading dimensions. Packed with no transpositions.
   int lda = M;
   int ldb = K;
   int ldc = M;

   // Tile using a 2D grid
   // blockIdx.x * blockDim.x + threadIdx.x ==> 0 - 255
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int mul = (blockIdx.x * blockDim.x + threadIdx.x);
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
   //printf("blockIdx.x = %d, blockDim.x = %d, threadIdx.x = %d, mul = %d, warpSize = %d, warpM = %d, warpN = %d\n",
   //       blockIdx.x, blockDim.x, threadIdx.x, mul, warpSize, warpM, warpN);

   /* 
   if (warpM > 0 || warpN > 0) {
      return;
   }
   */
 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

   wmma::fill_fragment(acc_frag, 0.0f);

   // Loop over k
   //int aRow = warpM * WMMA_M;
   //int bCol = warpN * WMMA_N;
   //printf("aRow = %d, warpM = %d, WMMA_M = %d, bCol = %d, warpN = %d, WMMA_N = %d\n", aRow, warpM, WMMA_M, bCol, warpN, WMMA_N);
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
         wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
         wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
      // Store the output
      wmma::store_matrix_sync(c + cRow + cCol * ldc, acc_frag, ldc, wmma::mem_col_major);
   }
}

void updatePair(Data data, Setting setting, float *c)
{
   for (int i = 0; i < data.numPairsToScore; i++)
   {
      Pair pair = data.d_PairsToScore[i];
      pair.score = c[getMemAddr(pair.docIdx, pair.reqIdx, data.numDocs, data.numReqs, COL_MAJOR)];
      data.d_rstWmma[i] = pair;
   }
}

void methodTensorSimple(Data data, Setting setting) {

   int MATRIX_M = data.numDocs;
   int MATRIX_N = data.numReqs;
   int MATRIX_K = data.embDim;

   T *a = data.d_doc;
   T *b = data.d_req;

   float *c_wmma;
   CHECK_CUDA(cudaMallocManaged(&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));

   PairBlock *pairBlock;
   CHECK_CUDA(cudaMallocManaged(&pairBlock, data.numPairsToScore * sizeof(PairBlock)));
   convertToPairBlock(data, pairBlock);
   int numPairBlocksToScore;
   dedupPairBlock(pairBlock, data.numPairsToScore, numPairBlocksToScore);
   
   // First: using WMMA
   dim3 gridDim;
   dim3 blockDim;
 
   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = 128;
   blockDim.y = 4;

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

   printf("\nRunning with wmma (simple)...\n");
   printf("M = %d, N = %d, K = %d.\n", MATRIX_M, MATRIX_N, MATRIX_K);
   cout << "blockDim: " << blockDim.x << " " << blockDim.y << endl;
   cout << "gridDim: " << gridDim.x << " " << gridDim.y << endl;
   CudaTimer timer;
   for (int t = -3; t < setting.numTrials; t++)
   {
      if (t == 0)
         timer.tic();
      tensorSimpleKernel <<< gridDim, blockDim >>> (a, b, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K);
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(cudaGetLastError());
   }
   cout << "wmma (simple) took " << timer.tocMs() / setting.numTrials << "ms" << endl;

   updatePair(data, setting, c_wmma);
}


