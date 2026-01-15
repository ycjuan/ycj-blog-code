#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <sstream>

template <typename T> struct CudaDeviceDeleter
{
    void operator()(T* ptr) const
    {
        cudaFree(ptr);
    }
};

template <typename T> struct CudaHostDeleter
{
    void operator()(T* ptr) const
    {
        cudaFreeHost(ptr);
    }
};


template <typename T> class CudaDeviceArray
{
public:
    CudaDeviceArray(uint64_t size, std::string name)
        : m_size(size)
        , m_name(name)
    {
        // --------------
        // Allocate device memory
        cudaError_t cudaError = cudaMalloc(&m_d_dataRawPtr, m_size * sizeof(T));
        if (cudaError != cudaSuccess)
        {
            std::ostringstream oss;
            oss << "Failed to allocate device memory for " << m_name << ": " << cudaGetErrorString(cudaError);
            throw std::runtime_error(oss.str());
        }

        // --------------
        // Create smart pointer
        m_d_dataSmartPtr.reset(m_d_dataRawPtr, CudaDeviceDeleter<T>());
    }

    // Accessors
    __device__ __host__ T* data() const { return m_d_dataRawPtr; }
    __device__ __host__ uint64_t getElementSize() const { return sizeof(T); }
    __device__ __host__ uint64_t getArraySize() const { return m_size; }
    __device__ __host__ uint64_t getArraySizeInBytes() const { return m_size * sizeof(T); }
    __device__ __host__ std::string getName() const { return m_name; }

private:
    uint64_t m_size;
    std::string m_name;
    T* m_d_dataRawPtr;
    std::shared_ptr<T> m_d_dataSmartPtr;
};



template <typename T> class CudaHostArray
{
public:
    CudaHostArray(uint64_t size, std::string name)
        : m_size(size)
        , m_name(name)
    {
        // --------------
        // Allocate device memory
        cudaError_t cudaError = cudaMallocHost(&m_h_dataRawPtr, m_size * sizeof(T));
        if (cudaError != cudaSuccess)
        {
            std::ostringstream oss;
            oss << "Failed to allocate host memory for " << m_name << ": " << cudaGetErrorString(cudaError);
            throw std::runtime_error(oss.str());
        }

        // --------------
        // Create smart pointer
        m_h_dataSmartPtr.reset(m_h_dataRawPtr, CudaHostDeleter<T>());
    }

    // Accessors
    __device__ __host__ T* data() const { return m_h_dataRawPtr; }
    __device__ __host__ uint64_t getElementSize() const { return sizeof(T); }
    __device__ __host__ uint64_t getArraySize() const { return m_size; }
    __device__ __host__ uint64_t getArraySizeInBytes() const { return m_size * sizeof(T); }
    __device__ __host__ std::string getName() const { return m_name; }

private:
    uint64_t m_size;
    std::string m_name;
    T* m_h_dataRawPtr;
    std::shared_ptr<T> m_h_dataSmartPtr;
};