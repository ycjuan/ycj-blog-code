#ifndef QUANT_DATA_STRUCT_CUH
#define QUANT_DATA_STRUCT_CUH

#include <sstream>
#include <iostream>

#include "common.cuh"

using namespace std;

// IMPORTANT: Do not change the following typedefs. These are only ones that works for WMMA.
typedef uint32_t T_QUANT;
typedef int32_t T_QUANT_RST;

struct QuantData
{
    int numDocs;
    int numReqs;
    int numInt32;
    T_QUANT *d_doc; // M=numDocs x N=numInt64
    T_QUANT *h_doc; 
    T_QUANT *d_req; // M=numReqs x N=numInt64
    T_QUANT *h_req;
    T_QUANT_RST *d_rstGpu; // M=numDocs x N=numReqs
    T_QUANT_RST *h_rstCpu;
    MemLayout docMemLayout = ROW_MAJOR; // IMPORTANT: Don't change this. a_frag in WMMA requires ROW_MAJOR
    MemLayout reqMemLayout = ROW_MAJOR; // IMPORTANT: Don't change this. b_frag in WMMA requires COL_MAJOR.
    // However, since the matrix here has a shape of (numReqs, numInt32), setting ROW_MAJOR here is equivalent to COL_MAJOR of a (numInt32, numReqs) matrix
    MemLayout rstLayoutCpu = ROW_MAJOR;
    MemLayout rstLayoutGpu = ROW_MAJOR;
    float timeMsGpu;

    void initRand(int numDocs, int numReqs, int numInt32);

    void free();

    void print();
};

void quantOpGpu(QuantData &data);

void quantOpCpu(QuantData &data);

#endif // QUANT_DATA_STRUCT_CUH