#ifndef DATA_STRUCT_CUH
#define DATA_STRUCT_CUH

#include <vector>
#include <cuda_fp16.h>
#include <sstream>
#include <cublas_v2.h>

#include "common.cuh"

struct AbmData
{
    int numDocs;
    int numReqs;
    int numClauses;
    int maxNumAttrs;

    std::vector<std::vector<std::vector<long>>> docAttr3D;
    std::vector<std::vector<std::vector<long>>> reqAttr3D;

    long *d_docAttr; // size = numDocs x maxNumAttrs
    long *h_docAttr;
    long *d_reqAttr; // size = numReqs x maxNumAttrs
    long *h_reqAttr;
    int *d_docOffset; // size = numDocs + 1
    int *h_docOffset;
    int *d_reqOffset; // size = numReqs + 1
    int *h_reqOffset;

    Pair *d_rstGpu;
    Pair *h_rstCpu;
    MemLayout docMemLayout;
    MemLayout reqMemLayout;
    const MemLayout rstMemLayout = COL_MAJOR; // CUBLAS always uses COL_MAJOR for the output matrix
    
    float timeMsGpu;
    float timeMsCpu;

    void initRand(int numDocs,
                  int numReqs,
                  std::vector<int> numAttrPerClause,
                  std::vector<int> cardinalityPerClause,
                  MemLayout docMemLayout,
                  MemLayout reqMemLayout);

    void free();

    void print();
};

#endif