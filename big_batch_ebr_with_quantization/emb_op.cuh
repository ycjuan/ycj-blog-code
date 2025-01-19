#ifndef GEMM_CUH
#define GEMM_CUH

#include "emb_data_struct.cuh"

void embOpGpu(EmbData &data);

void embOpCpu(EmbData &data);

#endif