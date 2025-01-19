#ifndef DATA_STRUCT_CUH
#define DATA_STRUCT_CUH

#include <cuda_fp16.h>
#include <sstream>
#include <cublas_v2.h>
#include "common.cuh"

typedef half T_EMB;
//typedef __nv_bfloat16 T_EMB;
//typedef float T_EMB;


struct EmbData
{
    int numDocs;
    int numReqs;
    int embDim;
    T_EMB *d_doc = nullptr; // M=numDocs x N=embDim
    T_EMB *h_doc = nullptr;
    T_EMB *d_req = nullptr; // M=numReqs x N=embDim
    T_EMB *h_req = nullptr;
    float *d_rst = nullptr; // M=numDocs x N=numReqs
    float *h_rst = nullptr;
    MemLayout docMemLayout;
    MemLayout reqMemLayout;
    const MemLayout rstMemLayout = COL_MAJOR; // CUBLAS always uses COL_MAJOR for the output matrix
    cublasHandle_t cublasHandle;
    Pair *d_mask = nullptr;
    Pair *h_mask = nullptr;
    size_t maskSize = 0;
    
    float timeMsCuBlas;
    float timeMsCpu;

    void initRand(int numDocs, int numReqs, int embDim, MemLayout docMemLayout, MemLayout reqMemLayout);

    void initRandMask(float passRate);

    void free();

    void print();
};

#endif