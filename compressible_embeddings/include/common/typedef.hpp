#pragma once

#include <cstdint>
#include <cuda_bf16.h>

typedef int32_t T_DOC_IDX; // Use signed integer as we need to use -1 to indicate an invalid index in caching.

typedef nv_bfloat16 T_EMB;

typedef uint64_t T_RQ;
