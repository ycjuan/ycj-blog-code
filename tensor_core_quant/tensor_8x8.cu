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

const int WMMA_M = 8;
const int WMMA_N = 8;
const int WMMA_K = 128;

__global__ void quantWmmaKernel(const unsigned *a, const unsigned *b, int *c, const unsigned M, const unsigned N, const unsigned K)
{
   using namespace nvcuda::wmma::experimental;
   int lda = K;
   int ldb = K;
   int ldc = N;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag0;
   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag1;
   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag2;
   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag3;
   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag4;
   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag5;
   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag6;
   wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag7;
   wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag0;
   wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag1;
   wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag2;
   wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag3;
   wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag4;
   wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag5;
   wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag6;
   wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag7;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag00;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag01;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag02;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag03;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag04;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag05;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag06;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag07;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag10;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag11;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag12;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag13;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag14;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag15;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag16;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag17;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag20;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag21;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag22;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag23;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag24;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag25;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag26;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag27;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag30;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag31;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag32;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag33;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag34;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag35;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag36;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag37;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag40;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag41;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag42;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag43;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag44;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag45;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag46;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag47;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag50;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag51;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag52;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag53;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag54;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag55;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag56;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag57;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag60;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag61;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag62;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag63;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag64;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag65;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag66;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag67;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag70;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag71;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag72;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag73;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag74;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag75;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag76;
   wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag77;
   wmma::fill_fragment(c_frag00, 0);
   wmma::fill_fragment(c_frag01, 0);
   wmma::fill_fragment(c_frag02, 0);
   wmma::fill_fragment(c_frag03, 0);
   wmma::fill_fragment(c_frag04, 0);
   wmma::fill_fragment(c_frag05, 0);
   wmma::fill_fragment(c_frag06, 0);
   wmma::fill_fragment(c_frag07, 0);
   wmma::fill_fragment(c_frag10, 0);
   wmma::fill_fragment(c_frag11, 0);
   wmma::fill_fragment(c_frag12, 0);
   wmma::fill_fragment(c_frag13, 0);
   wmma::fill_fragment(c_frag14, 0);
   wmma::fill_fragment(c_frag15, 0);
   wmma::fill_fragment(c_frag16, 0);
   wmma::fill_fragment(c_frag17, 0);
   wmma::fill_fragment(c_frag20, 0);
   wmma::fill_fragment(c_frag21, 0);
   wmma::fill_fragment(c_frag22, 0);
   wmma::fill_fragment(c_frag23, 0);
   wmma::fill_fragment(c_frag24, 0);
   wmma::fill_fragment(c_frag25, 0);
   wmma::fill_fragment(c_frag26, 0);
   wmma::fill_fragment(c_frag27, 0);
   wmma::fill_fragment(c_frag30, 0);
   wmma::fill_fragment(c_frag31, 0);
   wmma::fill_fragment(c_frag32, 0);
   wmma::fill_fragment(c_frag33, 0);
   wmma::fill_fragment(c_frag34, 0);
   wmma::fill_fragment(c_frag35, 0);
   wmma::fill_fragment(c_frag36, 0);
   wmma::fill_fragment(c_frag37, 0);
   wmma::fill_fragment(c_frag40, 0);
   wmma::fill_fragment(c_frag41, 0);
   wmma::fill_fragment(c_frag42, 0);
   wmma::fill_fragment(c_frag43, 0);
   wmma::fill_fragment(c_frag44, 0);
   wmma::fill_fragment(c_frag45, 0);
   wmma::fill_fragment(c_frag46, 0);
   wmma::fill_fragment(c_frag47, 0);
   wmma::fill_fragment(c_frag50, 0);
   wmma::fill_fragment(c_frag51, 0);
   wmma::fill_fragment(c_frag52, 0);
   wmma::fill_fragment(c_frag53, 0);
   wmma::fill_fragment(c_frag54, 0);
   wmma::fill_fragment(c_frag55, 0);
   wmma::fill_fragment(c_frag56, 0);
   wmma::fill_fragment(c_frag57, 0);
   wmma::fill_fragment(c_frag60, 0);
   wmma::fill_fragment(c_frag61, 0);
   wmma::fill_fragment(c_frag62, 0);
   wmma::fill_fragment(c_frag63, 0);
   wmma::fill_fragment(c_frag64, 0);
   wmma::fill_fragment(c_frag65, 0);
   wmma::fill_fragment(c_frag66, 0);
   wmma::fill_fragment(c_frag67, 0);
   wmma::fill_fragment(c_frag70, 0);
   wmma::fill_fragment(c_frag71, 0);
   wmma::fill_fragment(c_frag72, 0);
   wmma::fill_fragment(c_frag73, 0);
   wmma::fill_fragment(c_frag74, 0);
   wmma::fill_fragment(c_frag75, 0);
   wmma::fill_fragment(c_frag76, 0);
   wmma::fill_fragment(c_frag77, 0);

   int lda32 = lda / 32;
   int ldb32 = ldb / 32;
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
      size_t bCol0 = warpN * WMMA_N * 8;

      size_t bRow1 = i32;
      size_t bCol1 = bCol0 + WMMA_N;

      size_t bRow2 = i32;
      size_t bCol2 = bCol0 + WMMA_N * 2;

      size_t bRow3 = i32;
      size_t bCol3 = bCol0 + WMMA_N * 3;

      size_t bRow4 = i32;
      size_t bCol4 = bCol0 + WMMA_N * 4;

      size_t bRow5 = i32;
      size_t bCol5 = bCol0 + WMMA_N * 5;

      size_t bRow6 = i32;
      size_t bCol6 = bCol0 + WMMA_N * 6;

      size_t bRow7 = i32;
      size_t bCol7 = bCol0 + WMMA_N * 7;

      wmma::load_matrix_sync(a_frag0, a + aRow0 * lda32 + aCol0, lda);
      wmma::load_matrix_sync(a_frag1, a + aRow1 * lda32 + aCol1, lda);
      wmma::load_matrix_sync(a_frag2, a + aRow2 * lda32 + aCol2, lda);
      wmma::load_matrix_sync(a_frag3, a + aRow3 * lda32 + aCol3, lda);
      wmma::load_matrix_sync(a_frag4, a + aRow4 * lda32 + aCol4, lda);
      wmma::load_matrix_sync(a_frag5, a + aRow5 * lda32 + aCol5, lda);
      wmma::load_matrix_sync(a_frag6, a + aRow6 * lda32 + aCol6, lda);
      wmma::load_matrix_sync(a_frag7, a + aRow7 * lda32 + aCol7, lda);
      wmma::load_matrix_sync(b_frag0, b + bCol0 * ldb32 + bRow0, ldb);
      wmma::load_matrix_sync(b_frag1, b + bCol1 * ldb32 + bRow1, ldb);
      wmma::load_matrix_sync(b_frag2, b + bCol2 * ldb32 + bRow2, ldb);
      wmma::load_matrix_sync(b_frag3, b + bCol3 * ldb32 + bRow3, ldb);
      wmma::load_matrix_sync(b_frag4, b + bCol4 * ldb32 + bRow4, ldb);
      wmma::load_matrix_sync(b_frag5, b + bCol5 * ldb32 + bRow5, ldb);
      wmma::load_matrix_sync(b_frag6, b + bCol6 * ldb32 + bRow6, ldb);
      wmma::load_matrix_sync(b_frag7, b + bCol7 * ldb32 + bRow7, ldb);

      // Perform the matrix multiplication
      wmma::bmma_sync(c_frag00, a_frag0, b_frag0, c_frag00);
      wmma::bmma_sync(c_frag01, a_frag0, b_frag1, c_frag01);
      wmma::bmma_sync(c_frag02, a_frag0, b_frag2, c_frag02);
      wmma::bmma_sync(c_frag03, a_frag0, b_frag3, c_frag03);
      wmma::bmma_sync(c_frag04, a_frag0, b_frag4, c_frag04);
      wmma::bmma_sync(c_frag05, a_frag0, b_frag5, c_frag05);
      wmma::bmma_sync(c_frag06, a_frag0, b_frag6, c_frag06);
      wmma::bmma_sync(c_frag07, a_frag0, b_frag7, c_frag07);
      wmma::bmma_sync(c_frag10, a_frag1, b_frag0, c_frag10);
      wmma::bmma_sync(c_frag11, a_frag1, b_frag1, c_frag11);
      wmma::bmma_sync(c_frag12, a_frag1, b_frag2, c_frag12);
      wmma::bmma_sync(c_frag13, a_frag1, b_frag3, c_frag13);
      wmma::bmma_sync(c_frag14, a_frag1, b_frag4, c_frag14);
      wmma::bmma_sync(c_frag15, a_frag1, b_frag5, c_frag15);
      wmma::bmma_sync(c_frag16, a_frag1, b_frag6, c_frag16);
      wmma::bmma_sync(c_frag17, a_frag1, b_frag7, c_frag17);
      wmma::bmma_sync(c_frag20, a_frag2, b_frag0, c_frag20);
      wmma::bmma_sync(c_frag21, a_frag2, b_frag1, c_frag21);
      wmma::bmma_sync(c_frag22, a_frag2, b_frag2, c_frag22);
      wmma::bmma_sync(c_frag23, a_frag2, b_frag3, c_frag23);
      wmma::bmma_sync(c_frag24, a_frag2, b_frag4, c_frag24);
      wmma::bmma_sync(c_frag25, a_frag2, b_frag5, c_frag25);
      wmma::bmma_sync(c_frag26, a_frag2, b_frag6, c_frag26);
      wmma::bmma_sync(c_frag27, a_frag2, b_frag7, c_frag27);
      wmma::bmma_sync(c_frag30, a_frag3, b_frag0, c_frag30);
      wmma::bmma_sync(c_frag31, a_frag3, b_frag1, c_frag31);
      wmma::bmma_sync(c_frag32, a_frag3, b_frag2, c_frag32);
      wmma::bmma_sync(c_frag33, a_frag3, b_frag3, c_frag33);
      wmma::bmma_sync(c_frag34, a_frag3, b_frag4, c_frag34);
      wmma::bmma_sync(c_frag35, a_frag3, b_frag5, c_frag35);
      wmma::bmma_sync(c_frag36, a_frag3, b_frag6, c_frag36);
      wmma::bmma_sync(c_frag37, a_frag3, b_frag7, c_frag37);
      wmma::bmma_sync(c_frag40, a_frag4, b_frag0, c_frag40);
      wmma::bmma_sync(c_frag41, a_frag4, b_frag1, c_frag41);
      wmma::bmma_sync(c_frag42, a_frag4, b_frag2, c_frag42);
      wmma::bmma_sync(c_frag43, a_frag4, b_frag3, c_frag43);
      wmma::bmma_sync(c_frag44, a_frag4, b_frag4, c_frag44);
      wmma::bmma_sync(c_frag45, a_frag4, b_frag5, c_frag45);
      wmma::bmma_sync(c_frag46, a_frag4, b_frag6, c_frag46);
      wmma::bmma_sync(c_frag47, a_frag4, b_frag7, c_frag47);
      wmma::bmma_sync(c_frag50, a_frag5, b_frag0, c_frag50);
      wmma::bmma_sync(c_frag51, a_frag5, b_frag1, c_frag51);
      wmma::bmma_sync(c_frag52, a_frag5, b_frag2, c_frag52);
      wmma::bmma_sync(c_frag53, a_frag5, b_frag3, c_frag53);
      wmma::bmma_sync(c_frag54, a_frag5, b_frag4, c_frag54);
      wmma::bmma_sync(c_frag55, a_frag5, b_frag5, c_frag55);
      wmma::bmma_sync(c_frag56, a_frag5, b_frag6, c_frag56);
      wmma::bmma_sync(c_frag57, a_frag5, b_frag7, c_frag57);
      wmma::bmma_sync(c_frag60, a_frag6, b_frag0, c_frag60);
      wmma::bmma_sync(c_frag61, a_frag6, b_frag1, c_frag61);
      wmma::bmma_sync(c_frag62, a_frag6, b_frag2, c_frag62);
      wmma::bmma_sync(c_frag63, a_frag6, b_frag3, c_frag63);
      wmma::bmma_sync(c_frag64, a_frag6, b_frag4, c_frag64);
      wmma::bmma_sync(c_frag65, a_frag6, b_frag5, c_frag65);
      wmma::bmma_sync(c_frag66, a_frag6, b_frag6, c_frag66);
      wmma::bmma_sync(c_frag67, a_frag6, b_frag7, c_frag67);
      wmma::bmma_sync(c_frag70, a_frag7, b_frag0, c_frag70);
      wmma::bmma_sync(c_frag71, a_frag7, b_frag1, c_frag71);
      wmma::bmma_sync(c_frag72, a_frag7, b_frag2, c_frag72);
      wmma::bmma_sync(c_frag73, a_frag7, b_frag3, c_frag73);
      wmma::bmma_sync(c_frag74, a_frag7, b_frag4, c_frag74);
      wmma::bmma_sync(c_frag75, a_frag7, b_frag5, c_frag75);
      wmma::bmma_sync(c_frag76, a_frag7, b_frag6, c_frag76);
      wmma::bmma_sync(c_frag77, a_frag7, b_frag7, c_frag77);

   }

   int cRow0 = warpM * WMMA_M * 8;
   int cRow1 = cRow0 + WMMA_M * 1;
   int cRow2 = cRow0 + WMMA_M * 2;
   int cRow3 = cRow0 + WMMA_M * 3;
   int cRow4 = cRow0 + WMMA_M * 4;
   int cRow5 = cRow0 + WMMA_M * 5;
   int cRow6 = cRow0 + WMMA_M * 6;
   int cRow7 = cRow0 + WMMA_M * 7;

   int cCol0 = warpN * WMMA_N * 8;
   int cCol1 = cCol0 + WMMA_N * 1;
   int cCol2 = cCol0 + WMMA_N * 2;
   int cCol3 = cCol0 + WMMA_N * 3;
   int cCol4 = cCol0 + WMMA_N * 4;
   int cCol5 = cCol0 + WMMA_N * 5;
   int cCol6 = cCol0 + WMMA_N * 6;
   int cCol7 = cCol0 + WMMA_N * 7;

#pragma unroll
   for (int i = 0; i < c_frag00.num_elements; i++)
   {
      c_frag00.x[i] = K - c_frag00.x[i];
      c_frag01.x[i] = K - c_frag01.x[i];
      c_frag02.x[i] = K - c_frag02.x[i];
      c_frag03.x[i] = K - c_frag03.x[i];
      c_frag04.x[i] = K - c_frag04.x[i];
      c_frag05.x[i] = K - c_frag05.x[i];
      c_frag06.x[i] = K - c_frag06.x[i];
      c_frag07.x[i] = K - c_frag07.x[i];
      c_frag10.x[i] = K - c_frag10.x[i];
      c_frag11.x[i] = K - c_frag11.x[i];
      c_frag12.x[i] = K - c_frag12.x[i];
      c_frag13.x[i] = K - c_frag13.x[i];
      c_frag14.x[i] = K - c_frag14.x[i];
      c_frag15.x[i] = K - c_frag15.x[i];
      c_frag16.x[i] = K - c_frag16.x[i];
      c_frag17.x[i] = K - c_frag17.x[i];
      c_frag20.x[i] = K - c_frag20.x[i];
      c_frag21.x[i] = K - c_frag21.x[i];
      c_frag22.x[i] = K - c_frag22.x[i];
      c_frag23.x[i] = K - c_frag23.x[i];
      c_frag24.x[i] = K - c_frag24.x[i];
      c_frag25.x[i] = K - c_frag25.x[i];
      c_frag26.x[i] = K - c_frag26.x[i];
      c_frag27.x[i] = K - c_frag27.x[i];
      c_frag30.x[i] = K - c_frag30.x[i];
      c_frag31.x[i] = K - c_frag31.x[i];
      c_frag32.x[i] = K - c_frag32.x[i];
      c_frag33.x[i] = K - c_frag33.x[i];
      c_frag34.x[i] = K - c_frag34.x[i];
      c_frag35.x[i] = K - c_frag35.x[i];
      c_frag36.x[i] = K - c_frag36.x[i];
      c_frag37.x[i] = K - c_frag37.x[i];
      c_frag40.x[i] = K - c_frag40.x[i];
      c_frag41.x[i] = K - c_frag41.x[i];
      c_frag42.x[i] = K - c_frag42.x[i];
      c_frag43.x[i] = K - c_frag43.x[i];
      c_frag44.x[i] = K - c_frag44.x[i];
      c_frag45.x[i] = K - c_frag45.x[i];
      c_frag46.x[i] = K - c_frag46.x[i];
      c_frag47.x[i] = K - c_frag47.x[i];
      c_frag50.x[i] = K - c_frag50.x[i];
      c_frag51.x[i] = K - c_frag51.x[i];
      c_frag52.x[i] = K - c_frag52.x[i];
      c_frag53.x[i] = K - c_frag53.x[i];
      c_frag54.x[i] = K - c_frag54.x[i];
      c_frag55.x[i] = K - c_frag55.x[i];
      c_frag56.x[i] = K - c_frag56.x[i];
      c_frag57.x[i] = K - c_frag57.x[i];
      c_frag60.x[i] = K - c_frag60.x[i];
      c_frag61.x[i] = K - c_frag61.x[i];
      c_frag62.x[i] = K - c_frag62.x[i];
      c_frag63.x[i] = K - c_frag63.x[i];
      c_frag64.x[i] = K - c_frag64.x[i];
      c_frag65.x[i] = K - c_frag65.x[i];
      c_frag66.x[i] = K - c_frag66.x[i];
      c_frag67.x[i] = K - c_frag67.x[i];
      c_frag70.x[i] = K - c_frag70.x[i];
      c_frag71.x[i] = K - c_frag71.x[i];
      c_frag72.x[i] = K - c_frag72.x[i];
      c_frag73.x[i] = K - c_frag73.x[i];
      c_frag74.x[i] = K - c_frag74.x[i];
      c_frag75.x[i] = K - c_frag75.x[i];
      c_frag76.x[i] = K - c_frag76.x[i];
      c_frag77.x[i] = K - c_frag77.x[i];
   }

   wmma::store_matrix_sync(c + cRow0 * ldc + cCol0, c_frag00, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow0 * ldc + cCol1, c_frag01, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow0 * ldc + cCol2, c_frag02, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow0 * ldc + cCol3, c_frag03, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow0 * ldc + cCol4, c_frag04, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow0 * ldc + cCol5, c_frag05, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow0 * ldc + cCol6, c_frag06, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow0 * ldc + cCol7, c_frag07, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow1 * ldc + cCol0, c_frag10, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow1 * ldc + cCol1, c_frag11, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow1 * ldc + cCol2, c_frag12, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow1 * ldc + cCol3, c_frag13, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow1 * ldc + cCol4, c_frag14, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow1 * ldc + cCol5, c_frag15, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow1 * ldc + cCol6, c_frag16, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow1 * ldc + cCol7, c_frag17, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow2 * ldc + cCol0, c_frag20, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow2 * ldc + cCol1, c_frag21, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow2 * ldc + cCol2, c_frag22, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow2 * ldc + cCol3, c_frag23, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow2 * ldc + cCol4, c_frag24, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow2 * ldc + cCol5, c_frag25, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow2 * ldc + cCol6, c_frag26, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow2 * ldc + cCol7, c_frag27, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow3 * ldc + cCol0, c_frag30, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow3 * ldc + cCol1, c_frag31, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow3 * ldc + cCol2, c_frag32, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow3 * ldc + cCol3, c_frag33, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow3 * ldc + cCol4, c_frag34, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow3 * ldc + cCol5, c_frag35, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow3 * ldc + cCol6, c_frag36, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow3 * ldc + cCol7, c_frag37, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow4 * ldc + cCol0, c_frag40, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow4 * ldc + cCol1, c_frag41, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow4 * ldc + cCol2, c_frag42, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow4 * ldc + cCol3, c_frag43, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow4 * ldc + cCol4, c_frag44, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow4 * ldc + cCol5, c_frag45, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow4 * ldc + cCol6, c_frag46, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow4 * ldc + cCol7, c_frag47, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow5 * ldc + cCol0, c_frag50, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow5 * ldc + cCol1, c_frag51, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow5 * ldc + cCol2, c_frag52, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow5 * ldc + cCol3, c_frag53, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow5 * ldc + cCol4, c_frag54, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow5 * ldc + cCol5, c_frag55, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow5 * ldc + cCol6, c_frag56, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow5 * ldc + cCol7, c_frag57, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow6 * ldc + cCol0, c_frag60, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow6 * ldc + cCol1, c_frag61, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow6 * ldc + cCol2, c_frag62, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow6 * ldc + cCol3, c_frag63, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow6 * ldc + cCol4, c_frag64, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow6 * ldc + cCol5, c_frag65, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow6 * ldc + cCol6, c_frag66, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow6 * ldc + cCol7, c_frag67, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow7 * ldc + cCol0, c_frag70, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow7 * ldc + cCol1, c_frag71, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow7 * ldc + cCol2, c_frag72, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow7 * ldc + cCol3, c_frag73, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow7 * ldc + cCol4, c_frag74, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow7 * ldc + cCol5, c_frag75, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow7 * ldc + cCol6, c_frag76, ldc, wmma::mem_row_major);
   wmma::store_matrix_sync(c + cRow7 * ldc + cCol7, c_frag77, ldc, wmma::mem_row_major);   
   
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

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32) / 8;
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y) / 8;

   cout << "blockDim: " << blockDim.x << " " << blockDim.y << endl;
   cout << "gridDim: " << gridDim.x << " " << gridDim.y << endl;

   printf("Running with wmma...\n");
   CudaTimer timer;
   for (int t = -3; t < setting.kNumTrials; t++)
   {
      if (t == 0)
         timer.tic();
      quantWmmaKernel <<< gridDim, blockDim >>> (a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K * 32);
      cudaErrCheck(cudaDeviceSynchronize());
      cudaErrCheck(cudaGetLastError());
   }
   cout << "wmma took " << timer.tocMs() / setting.kNumTrials << "ms" << endl;
}


