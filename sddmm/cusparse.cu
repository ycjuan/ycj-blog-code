/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

/* The code is modified from:
 *   https://github.com/NVIDIA/CUDALibrarySamples/tree/467734659975dd2d795609bd7c01930cc560338f/cuSPARSE/sddmm_csr
 */ 

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include "common.cuh"
#include "util.cuh"

using namespace std;

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

#define CHECK_CUSPARSE(func)                                                     \
    {                                                                            \
        cusparseStatus_t status = (func);                                        \
        if (status != CUSPARSE_STATUS_SUCCESS)                                   \
        {                                                                        \
            string error = "CUSPARSE API failed at line " + to_string(__LINE__); \
            throw runtime_error(error);                                          \
        }                                                                        \
    }

void methodCusparse(Data data, Setting setting) 
{
    // Host problem definition
    int   A_num_rows   = data.numDocs;
    int   A_num_cols   = data.embDim;
    int   B_num_rows   = A_num_cols;
    int   B_num_cols   = data.numReqs;
    int   C_nnz        = data.numPairsToScore;
    int   lda          = (data.docMemLayout == ROW_MAJOR)? A_num_cols : A_num_rows;
    int   ldb          = (data.reqMemLayout == COL_MAJOR)? B_num_cols : B_num_rows;
    int   A_size       = (data.docMemLayout == ROW_MAJOR)? lda * A_num_rows : lda * A_num_cols;
    int   B_size       = (data.reqMemLayout == COL_MAJOR)? ldb * B_num_rows : ldb * B_num_cols;
    float alpha        = 1.0f;
    float beta         = 0.0f;
    cout << "lda = " << lda << ", ldb = " << ldb << endl;
    //--------------------------------------------------------------------------
    int   *hC_offsets, *hC_columns;
    T *hB, *hA;
    float *hC_values;
    // data.d_doc is allocated by cudaMallocManaged. however, it seems it doesn't work for cusparse for some reason..
    // that's why I'm treating it as "host memory", and then copy to dA which is allocated by cudaMalloc.
    hA = data.d_doc;
    hB = data.d_req;
    CHECK_CUDA( cudaMallocHost((void**) &hC_offsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMallocHost((void**) &hC_columns, C_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMallocHost((void**) &hC_values,  C_nnz * sizeof(float)) )
    
    #pragma omp parallel for
    for (size_t pairIdx = 0; pairIdx < data.numPairsToScore; pairIdx++)
        data.d_PairsToScore[pairIdx].score = 0;

    coo2Csr(data, hC_offsets, hC_columns, hC_values);

    // Device memory management
    int   *dC_offsets, *dC_columns;
    T *dB, *dA;
    float *dC_values;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_offsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_offsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_columns, C_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_values,  C_nnz * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size * sizeof(float),
                           cudaMemcpyDeviceToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float),
                           cudaMemcpyDeviceToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_offsets, hC_offsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_columns, hC_columns, C_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_values, hC_values, C_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )

    cudaDeviceSynchronize();

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseDnMatDescr_t matA, matB;
    cusparseSpMatDescr_t matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create dense matrix A
    cusparseOrder_t orderA = (data.docMemLayout == ROW_MAJOR)? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
    CHECK_CUSPARSE( cusparseCreateDnMat(&matA, A_num_rows, A_num_cols, lda, dA,
                                        CUDA_R_32F, orderA) )
    // Create dense matrix B
    cusparseOrder_t orderB = (data.reqMemLayout == COL_MAJOR)? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, orderB) )
    // Create sparse matrix C in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, A_num_rows, B_num_cols, C_nnz,
                                      dC_offsets, dC_columns, dC_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSDDMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute preprocess (optional)
    /*
    The doc says:
        The function cusparseSDDMM_preprocess() can be called before cusparseSDDMM to speedup the actual computation. 
        It is useful when cusparseSDDMM is called multiple times with the same sparsity pattern (matC). 
        The values of the dense matrices (matA, matB) can change arbitrarily.
    Since we can't assume "the same sparsity pattern (matC)" for our application, we don't use this function here.
    */
    /*
    CHECK_CUSPARSE( cusparseSDDMM_preprocess(
                                  handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                  CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer) )
    */
    // execute SpMM
    Timer timer;
    timer.tic();
    for (int t = -3; t < setting.numTrials; t++)
    {
        /*
        also tried to use this, but doesn't have any effect on the performance.
        CHECK_CUSPARSE(cusparseSDDMM_preprocess(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer))
        */
        CHECK_CUSPARSE(cusparseSDDMM(handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                     CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer))
        CHECK_CUDA( cudaDeviceSynchronize() )
    }
    cout << "CUSPARSE time: " << timer.tocMs() / setting.numTrials << " ms" << endl;
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroyDnMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    
    CHECK_CUDA( cudaMemcpy(hC_values, dC_values, C_nnz * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    size_t pairIdx = 0;
    for (int docIdx = 0; docIdx < data.numDocs; docIdx++)
    {
        int start = hC_offsets[docIdx];
        int end = hC_offsets[docIdx + 1];
        for (int i = start; i < end; i++)
        {
            Pair pair;
            pair.reqIdx = hC_columns[i];
            pair.docIdx = docIdx;
            pair.score = hC_values[i];
            data.d_rstCusparse[pairIdx++] = pair;
        }
    }

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC_offsets) )
    CHECK_CUDA( cudaFree(dC_columns) )
    CHECK_CUDA( cudaFree(dC_values) )
    CHECK_CUDA( cudaFreeHost(hC_offsets) )
    CHECK_CUDA( cudaFreeHost(hC_columns) )
    CHECK_CUDA( cudaFreeHost(hC_values) )
}
