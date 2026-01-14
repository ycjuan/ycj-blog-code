#ifndef LIB_GPU_RETRIEVAL_DATA_STRUCT_CUDA_ARRAY_CUH
#define LIB_GPU_RETRIEVAL_DATA_STRUCT_CUDA_ARRAY_CUH

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <iostream>

namespace libgpuretrieval::data_struct
{
    template <typename T>
    struct CudaDeleter
    {
        CudaDeleter(T **pp_raw, bool isHost) : pp_raw_(pp_raw), isHost_(isHost) {}

        void operator()(T *ptr) const
        {
            if (ptr)
            {
                if (isHost_)
                {
                    cudaFreeHost(ptr);
                }
                else
                {
                    cudaFree(ptr);
                }

                if (*pp_raw_ == ptr)
                {
                    *pp_raw_ = nullptr;
                }
            }
        }

    private:
        T **const pp_raw_;
        const bool isHost_;
    };

    template <typename T>
    class CudaArray
    {
    public:
        // Constructor
        CudaArray() : kSize_(0), kIsHost_(false), d_data_raw_(nullptr) { }
        CudaArray(uint64_t size, bool isHost = false)
            : kSize_(size), kIsHost_(isHost)
        {
            cudaError_t cudaError; 
            
            if (isHost)
            {
                cudaError = cudaMallocHost(&d_data_raw_, kSize_ * sizeof(T));
            }
            else
            {
                cudaError = cudaMalloc(&d_data_raw_, kSize_ * sizeof(T));
            }

            if (cudaError != cudaSuccess)
            {
                throw std::runtime_error("Failed to allocate device memory for CudaArray: " + std::string(cudaGetErrorString(cudaError)));
            }
            d_data_.reset(d_data_raw_, CudaDeleter<T>(&d_data_raw_, kIsHost_));
        }

        // Accessors
        __device__ __host__ T *data() const { return d_data_raw_; }
        __device__ __host__ uint64_t size() const { return kSize_; }
        __device__ __host__ bool isHost() const { return kIsHost_; }

    private:
        uint64_t kSize_;
        bool kIsHost_;
        T *d_data_raw_;
        std::shared_ptr<T> d_data_;
    };
}

#endif // LIB_GPU_RETRIEVAL_DATA_STRUCT_CUDA_ARRAY_CUH