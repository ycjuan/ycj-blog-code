#pragma once

#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>

constexpr bool kPrintDebug = true;

template <typename T>
struct CudaNullDeleter
{
    void operator()(T* /*ptr*/) const { }
};

// Deleter that signals release by setting *isReleased = true instead of freeing
// memory. Optionally calls onRelease() after signalling — used by
// UniversalDeviceBuffer to notify waiting threads (e.g. condition variable)
// when a slice is returned.
template <typename T>
struct CudaReleaseSignalDeleter
{
    std::shared_ptr<bool> m_isReleased;
    std::function<void()> m_onRelease;
    CudaReleaseSignalDeleter(std::shared_ptr<bool> isReleased, std::function<void()> onRelease = nullptr)
        : m_isReleased(std::move(isReleased))
        , m_onRelease(std::move(onRelease))
    {
    }
    void operator()(T* /*ptr*/) const
    {
        *m_isReleased = true;
        if (m_onRelease)
            m_onRelease();
    }
};

template <typename T>
struct CudaDeviceDeleter
{
    void operator()(T* ptr) const
    {
        cudaFree(ptr);
        if (kPrintDebug)
        {
            std::cout << "cudaFree(" << (void*)ptr << ")" << std::endl;
        }
    }
};

template <typename T>
struct CudaHostDeleter
{
    void operator()(T* ptr) const
    {
        cudaFreeHost(ptr);
        if (kPrintDebug)
        {
            std::cout << "cudaFreeHost(" << (void*)ptr << ")" << std::endl;
        }
    }
};

template <typename T>
class CudaArray
{
public:
    // See https://leimao.github.io/blog/CPP-Base-Class-Destructors/ for why we
    // need to make this virtual
    virtual ~CudaArray() = default;

    // Accessors
    __device__ __host__ T*       data() const { return m_p_dataRawPtr; }
    __device__ __host__ uint64_t getElementSize() const { return sizeof(T); }
    __device__ __host__ uint64_t getArraySize() const { return m_size; }
    __device__ __host__ uint64_t getArraySizeInBytes() const { return m_size * sizeof(T); }
    __host__ std::string getName() const { return m_name; }

protected: // Making the constructor protected will make this class
           // non-instantiable to the users
    CudaArray(uint64_t size, std::string name)
        : m_size(size)
        , m_name(name)
    {
    }

    // Constructor for wrapping an externally managed pointer. The shared_ptr uses
    // a no-op deleter so it never frees the memory — the caller owns the
    // lifetime.
    CudaArray(T* ptr, uint64_t size, std::string name)
        : m_size(size)
        , m_name(name)
        , m_p_dataRawPtr(ptr)
        , m_p_dataSmartPtr(ptr, CudaNullDeleter<T>())
    {
    }

    // Constructor for wrapping a slice of a larger buffer. When the last copy
    // goes out of scope, the deleter sets *isReleased = true and calls
    // onRelease() (if provided) so the owning buffer can reclaim the segment and
    // wake waiting threads.
    CudaArray(T*                    ptr,
              uint64_t              size,
              std::string           name,
              std::shared_ptr<bool> isReleased,
              std::function<void()> onRelease = nullptr)
        : m_size(size)
        , m_name(name)
        , m_p_dataRawPtr(ptr)
        , m_p_dataSmartPtr(ptr, CudaReleaseSignalDeleter<T>(std::move(isReleased), std::move(onRelease)))
    {
    }

    uint64_t    m_size;
    std::string m_name;
    // We need a separate raw pointer (instead of just calling
    // m_p_dataSmartPtr.get()) because shared_ptr cannot be used in device code.
    // m_p_dataRawPtr is what gets passed into CUDA kernels.
    T* m_p_dataRawPtr = nullptr;

    // It is very important to note that we use shared_ptr here to release memory,
    // instead of just call cudaFree / cudaFreeHost in the destructor. This is
    // because we want to make this class copyable, but when we copy, we only want
    // to copy the pointer, not the entire memory. This implies that we need some
    // reference counting mechanism to release the memory when the last copy is
    // out of scope. shared_ptr is a good choise for this purpose.
    std::shared_ptr<T> m_p_dataSmartPtr;
};

template <typename T>
class CudaDeviceArray : public CudaArray<T>
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
            std::cout << "cudaMalloc(" << (void*)this->m_p_dataRawPtr << ", " << this->m_size * sizeof(T) << ")"
                      << std::endl;
        }

        // --------------
        // Create smart pointer
        this->m_p_dataSmartPtr.reset(this->m_p_dataRawPtr, CudaDeviceDeleter<T>());
    }

    CudaDeviceArray(T* ptr, uint64_t size, std::string name)
        : CudaArray<T>(ptr, size, name)
    {
    }

    CudaDeviceArray(T*                    ptr,
                    uint64_t              size,
                    std::string           name,
                    std::shared_ptr<bool> isReleased,
                    std::function<void()> onRelease = nullptr)
        : CudaArray<T>(ptr, size, name, std::move(isReleased), std::move(onRelease))
    {
    }
};

template <typename T>
class CudaHostArray : public CudaArray<T>
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
        if (kPrintDebug)
        {
            std::cout << "cudaMallocHost(" << (void*)this->m_p_dataRawPtr << ", " << this->m_size * sizeof(T) << ")"
                      << std::endl;
        }

        // --------------
        // Create smart pointer
        this->m_p_dataSmartPtr.reset(this->m_p_dataRawPtr, CudaHostDeleter<T>());
    }

    CudaHostArray(T* ptr, uint64_t size, std::string name)
        : CudaArray<T>(ptr, size, name)
    {
    }

    CudaHostArray(T*                    ptr,
                  uint64_t              size,
                  std::string           name,
                  std::shared_ptr<bool> isReleased,
                  std::function<void()> onRelease = nullptr)
        : CudaArray<T>(ptr, size, name, std::move(isReleased), std::move(onRelease))
    {
    }
};