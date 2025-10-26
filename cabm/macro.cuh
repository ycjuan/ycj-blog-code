#pragma once

#define CHECK_CUDA(func)                                                                                               \
    {                                                                                                                  \
        cudaError_t status = (func);                                                                                   \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            std::string error = "CUDA API failed at line " + std::to_string(__LINE__)                                  \
                + " with error: " + cudaGetErrorString(status) + "\n";                                                 \
            throw std::runtime_error(error);                                                                           \
        }                                                                                                              \
    }
