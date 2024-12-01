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
const int BLOCK_X = 32;
const int BLOCK_Y = 1;
const int WARP_SIZE = 32;
const int PAIR_BLOCK_SIZE_X = BLOCK_X * WMMA_M / WARP_SIZE;
const int PAIR_BLOCK_SIZE_Y = BLOCK_Y * WMMA_N;

void convertToPairBlock(Data data, PairBlock *pairBlock)
{
   cout << "PAIR_BLOCK_SIZE_X: " << PAIR_BLOCK_SIZE_X << ", PAIR_BLOCK_SIZE_Y: " << PAIR_BLOCK_SIZE_Y << endl;
   for (int i = 0; i < data.numPairsToScore; i++)
   {
      Pair pair = data.d_PairsToScore[i];
      pairBlock[i].docIdx = pair.docIdx;
      pairBlock[i].reqIdx = pair.reqIdx;
      pairBlock[i].docBlockIdx = pair.docIdx / PAIR_BLOCK_SIZE_X;
      pairBlock[i].docBlockOffset = pair.docIdx % PAIR_BLOCK_SIZE_X;
      pairBlock[i].reqBlockIdx = pair.reqIdx / PAIR_BLOCK_SIZE_Y;
      pairBlock[i].reqBlockOffset = pair.reqIdx % PAIR_BLOCK_SIZE_Y;
   }
}

void dedupPairBlock(PairBlock *pairBlock, int numPairsToScore, int &numPairBlocks)
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
         pairBlock[++numPairBlocks] = pair;
         currPair = pair;
      }
   }
}

template <typename WMMA_A_MEM_LAYOUT, typename WMMA_B_MEM_LAYOUT>
__global__ void tensorSimpleKernel(
   T *a, T *b, float *c, int M, int N, int K, Setting setting, int numPairBlocks, PairBlock *d_pairBlock) {
         wmma::layout_t c_layout = setting.wmmaOutputMemLayout == ROW_MAJOR ? wmma::mem_row_major : wmma::mem_col_major;
   int lda = setting.docMemLayout == ROW_MAJOR ? K : M;
   int ldb = setting.reqMemLayout == ROW_MAJOR ? K : N;
   int ldc = setting.wmmaOutputMemLayout == ROW_MAJOR ? N : M;

   // Tile using a 2D grid
   // blockIdx.x * blockDim.x + threadIdx.x ==> 0 - 255
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
   //int mul = (blockIdx.x * blockDim.x + threadIdx.x);
   //int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
   //printf("blockIdx.x = %d, blockDim.x = %d, threadIdx.x = %d, mul = %d, warpSize = %d, warpM = %d, warpN = %d\n",
   //       blockIdx.x, blockDim.x, threadIdx.x, mul, warpSize, warpM, warpN);

   /* 
   if (warpM > 0 || warpN > 0) {
      return;
   }
   */
 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, WMMA_A_MEM_LAYOUT> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, WMMA_B_MEM_LAYOUT> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   //wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

   wmma::fill_fragment(acc_frag, 0.0f);

   // Loop over k
   //int aRow = warpM * WMMA_M;
   //int bCol = warpN * WMMA_N;
   //printf("aRow = %d, warpM = %d, WMMA_M = %d, bCol = %d, warpN = %d, WMMA_N = %d\n", aRow, warpM, WMMA_M, bCol, warpN, WMMA_N);
   PairBlock pairBlock = d_pairBlock[warpM];
   int row = pairBlock.docBlockIdx * PAIR_BLOCK_SIZE_X;
   int col = pairBlock.reqBlockIdx * PAIR_BLOCK_SIZE_Y;
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = row;
      int aCol = i;

      int bRow = i;
      int bCol = col;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
         size_t addrA = getMemAddr(aRow, aCol, M, K, setting.docMemLayout);
         size_t addrB = getMemAddr(bCol, bRow, N, K, setting.reqMemLayout);
         wmma::load_matrix_sync(a_frag, a + addrA, lda);
         wmma::load_matrix_sync(b_frag, b + addrB, ldb);

         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = row;
   int cCol = col;

   if (cRow < M && cCol < N) {
      // Store the output
      wmma::store_matrix_sync(c + cRow + cCol * ldc, acc_frag, ldc, c_layout);
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

#pragma omp parallel for
   for (size_t pairIdx = 0; pairIdx < data.numPairsToScore; pairIdx++)
      data.d_PairsToScore[pairIdx].score = 0;

   if (setting.reqFirst)
      sort(data.d_PairsToScore, data.d_PairsToScore + data.numPairsToScore, pairComparatorReqFirst);
   else
      sort(data.d_PairsToScore, data.d_PairsToScore + data.numPairsToScore, pairComparatorDocFirst);

   PairBlock *pairBlock;
   CHECK_CUDA(cudaMallocManaged(&pairBlock, data.numPairsToScore * sizeof(PairBlock)));
   convertToPairBlock(data, pairBlock);
   int numPairBlocksToScore;
   dedupPairBlock(pairBlock, data.numPairsToScore, numPairBlocksToScore);

   cout << "numPairsToScore: " << data.numPairsToScore
        << ", numPairBlocksToScore: " << numPairBlocksToScore << endl;

   // First: using WMMA
   dim3 gridDim;
   dim3 blockDim;
 
   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = BLOCK_X;
   blockDim.y = BLOCK_Y;

   gridDim.x = numPairBlocksToScore * 32;
   gridDim.y = 1;//(MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

   printf("\nRunning with wmma (simple)...\n");
   printf("M = %d, N = %d, K = %d.\n", MATRIX_M, MATRIX_N, MATRIX_K);
   cout << "blockDim: " << blockDim.x << " " << blockDim.y << endl;
   cout << "gridDim: " << gridDim.x << " " << gridDim.y << endl;
   CudaTimer timer;
   for (int t = -3; t < setting.numTrials; t++)
   {
      if (t == 0)
         timer.tic();

      if (data.docMemLayout == ROW_MAJOR && data.reqMemLayout == ROW_MAJOR)
         tensorSimpleKernel<wmma::row_major, wmma::col_major>
             <<<gridDim, blockDim>>>(a, b, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, setting, numPairBlocksToScore, pairBlock);
      else if (data.docMemLayout == ROW_MAJOR && data.reqMemLayout == COL_MAJOR)
         tensorSimpleKernel<wmma::row_major, wmma::row_major>
             <<<gridDim, blockDim>>>(a, b, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, setting, numPairBlocksToScore, pairBlock);
      else if (data.docMemLayout == COL_MAJOR && data.reqMemLayout == ROW_MAJOR)
         tensorSimpleKernel<wmma::col_major, wmma::col_major>
             <<<gridDim, blockDim>>>(a, b, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, setting, numPairBlocksToScore, pairBlock);
      else
         tensorSimpleKernel<wmma::col_major, wmma::row_major>
             <<<gridDim, blockDim>>>(a, b, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, setting, numPairBlocksToScore, pairBlock);

      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(cudaGetLastError());
   }
   cout << "wmma (simple) took " << timer.tocMs() / setting.numTrials << "ms" << endl;

   updatePair(data, setting, c_wmma);
}


