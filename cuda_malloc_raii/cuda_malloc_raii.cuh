#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <iostream>

constexpr bool kPrintDebug = true;

template <typename T> struct CudaDeviceDeleter
{
    void operator()(T* ptr) const 
    { 
        cudaFree(ptr); 
        if (kPrintDebug)
        {
            std::cout << "cudaFree(" << ptr << ")" << std::endl;
        }
    }
};

template <typename T> struct CudaHostDeleter
{
    void operator()(T* ptr) const 
    { 
        cudaFreeHost(ptr); 
        if (kPrintDebug)
        {
            std::cout << "cudaFreeHost(" << ptr << ")" << std::endl;
        }
    }
};

template <typename T> class CudaArray
{
public:
    // See https://leimao.github.io/blog/CPP-Base-Class-Destructors/ for why we make this virtual
    virtual ~CudaArray() = default;

    // Accessors
    __device__ __host__ T* data() const { return m_p_dataRawPtr; }
    __device__ __host__ uint64_t getElementSize() const { return sizeof(T); }
    __device__ __host__ uint64_t getArraySize() const { return m_size; }
    __device__ __host__ uint64_t getArraySizeInBytes() const { return m_size * sizeof(T); }
    __device__ __host__ std::string getName() const { return m_name; }

protected: // Making the constructor protected will make this class non-instantiable to the users
    CudaArray(uint64_t size, std::string name)
        : m_size(size)
        , m_name(name)
    {
    }

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
        if (kPrintDebug)
        {
            std::cout << "cudaMalloc(" << this->m_p_dataRawPtr << ", " << this->m_size * sizeof(T) << ")" << std::endl;
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