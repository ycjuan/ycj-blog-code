#pragma once

#include <cuda_bf16.h>

// Embedding element type. bfloat16 halves memory bandwidth vs float32 with
// minimal accuracy loss for dot-product similarity.
typedef nv_bfloat16 T_EMB;
