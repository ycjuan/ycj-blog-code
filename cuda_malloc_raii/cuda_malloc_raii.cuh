#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>

template <typename T> struct CudaDeviceDeleter
{
    void operator()(T* ptr) const { cudaFree(ptr); }
};

template <typename T> struct CudaHostDeleter
{
    void operator()(T* ptr) const { cudaFreeHost(ptr); }
};

template <typename T> class CudaArray
{
public:
    CudaArray(uint64_t size, std::string name)
        : m_size(size)
        , m_name(name)
    {
    }

    // virtual ~CudaArray() = default; // Very important to have this, see
    // https://leimao.github.io/blog/CPP-Base-Class-Destructors/ for the reason

    // Accessors
    __device__ __host__ T* data() const { return m_p_dataRawPtr; }
    __device__ __host__ uint64_t getElementSize() const { return sizeof(T); }
    __device__ __host__ uint64_t getArraySize() const { return m_size; }
    __device__ __host__ uint64_t getArraySizeInBytes() const { return m_size * sizeof(T); }
    __device__ __host__ std::string getName() const { return m_name; }

protected:
    uint64_t m_size;
    std::string m_name;
    T* m_p_dataRawPtr;
    std::shared_ptr<T> m_p_dataSmartPtr;
};

template <typename T> class CudaDeviceArray : public CudaArray<T>
{
public:
    CudaDeviceArray(uint64_t size, std::string name)
        : CudaArray<T>(size, name)
    {
        // --------------
        // Allocate device memory
        cudaError_t cudaError = cudaMalloc(&this->m_p_dataRawPtr, this->m_size * sizeof(T));
        if (cudaError != cudaSuccess)
        {
            std::ostringstream oss;
            oss << "Failed to allocate device memory for " << this->m_name << ": " << cudaGetErrorString(cudaError);
            throw std::runtime_error(oss.str());
        }

        // --------------
        // Create smart pointer
        this->m_p_dataSmartPtr.reset(this->m_p_dataRawPtr, CudaDeviceDeleter<T>());
    }
};

template <typename T> class CudaHostArray : public CudaArray<T>
{
public:
    CudaHostArray(uint64_t size, std::string name)
        : CudaArray<T>(size, name)
    {

        // --------------
        // Allocate host memory
        cudaError_t cudaError = cudaMallocHost(&this->m_p_dataRawPtr, this->m_size * sizeof(T));
        if (cudaError != cudaSuccess)
        {
            std::ostringstream oss;
            oss << "Failed to allocate host memory for " << this->m_name << ": " << cudaGetErrorString(cudaError);
            throw std::runtime_error(oss.str());
        }

        // --------------
        // Create smart pointer
        this->m_p_dataSmartPtr.reset(this->m_p_dataRawPtr, CudaHostDeleter<T>());
    }
};