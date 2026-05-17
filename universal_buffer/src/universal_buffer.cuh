#pragma once

#include "cuda_malloc_raii.cuh"
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <vector>

enum class OomPolicy
{
    kThrow,    // throw std::runtime_error immediately if no space (default)
    kWaitSome, // block until at least one slice is returned, then retry
    kWaitAll,  // block until all slices are returned, grow the buffer, then retry
};

struct MemSegment
{
    uint64_t addrBeginIncl;        // byte offset from buffer base, inclusive
    uint64_t addrEndExcl;          // byte offset from buffer base, exclusive
    std::shared_ptr<bool> isReleased;
};

class UniversalDeviceBuffer
{
public:
    UniversalDeviceBuffer(uint64_t sizeInBytes, std::string name,
                          OomPolicy oomPolicy = OomPolicy::kThrow)
        : m_buffer(sizeInBytes, name)
        , m_name(name)
        , m_usedBytes(0)
        , m_oomPolicy(oomPolicy)
    {
    }

    // Find a free segment of sizeInBytes and return a non-owning CudaDeviceArray<char> wrapping it.
    // Behaviour on OOM is controlled by the OomPolicy set at construction.
    CudaDeviceArray<char> getBuffer(uint64_t sizeInBytes)
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        while (true)
        {
            pruneReleasedSegments();

            uint64_t offset;
            bool found = tryFindFreeOffset(sizeInBytes, offset);

            if (found)
            {
                auto isReleased = std::make_shared<bool>(false);
                auto onRelease  = [this]() { m_cv.notify_all(); };
                m_usedSegments.push_back({offset, offset + sizeInBytes, isReleased});
                std::sort(m_usedSegments.begin(), m_usedSegments.end(),
                          [](const MemSegment& a, const MemSegment& b)
                          { return a.addrBeginIncl < b.addrBeginIncl; });
                m_usedBytes += sizeInBytes;
                char* ptr = m_buffer.data() + offset;
                return CudaDeviceArray<char>(ptr, sizeInBytes, "", isReleased, onRelease);
            }

            if (m_oomPolicy == OomPolicy::kThrow)
            {
                throw std::runtime_error("UniversalDeviceBuffer: no free contiguous space of " +
                                         std::to_string(sizeInBytes) + " bytes");
            }
            else if (m_oomPolicy == OomPolicy::kWaitSome)
            {
                m_cv.wait(lock); // wait for any slice to be returned
            }
            else // kWaitAll
            {
                // Wait until all slices are returned, then grow the buffer
                m_cv.wait(lock, [this]() { pruneReleasedSegments(); return m_usedSegments.empty(); });
                uint64_t newSize = static_cast<uint64_t>(m_buffer.getArraySize() * m_kGrowthFactor);
                m_buffer = CudaDeviceArray<char>(newSize, m_name);
                m_usedBytes = 0;
            }
        }
    }

    uint64_t getTotalBytes() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_buffer.getArraySize();
    }
    uint64_t getFreeBytes()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        pruneReleasedSegments();
        return m_buffer.getArraySize() - m_usedBytes;
    }

private:
    // CUDA requires data to be aligned to 256 bytes for optimal memory access.
    // See https://leimao.github.io/blog/CUDA-Data-Alignment/
    static constexpr uint64_t m_kMemAlignmentInByte = 256;
    static constexpr double   m_kGrowthFactor        = 1.1;

    mutable std::mutex      m_mutex;
    std::condition_variable m_cv;
    CudaDeviceArray<char>   m_buffer;
    std::string             m_name;
    std::vector<MemSegment> m_usedSegments;
    uint64_t                m_usedBytes;
    OomPolicy               m_oomPolicy;

    void pruneReleasedSegments()
    {
        for (const MemSegment& seg : m_usedSegments)
            if (*seg.isReleased)
                m_usedBytes -= (seg.addrEndExcl - seg.addrBeginIncl);

        m_usedSegments.erase(
            std::remove_if(m_usedSegments.begin(), m_usedSegments.end(),
                           [](const MemSegment& s) { return *s.isReleased; }),
            m_usedSegments.end());
    }

    static uint64_t alignUp(uint64_t offset)
    {
        return (offset + m_kMemAlignmentInByte - 1) / m_kMemAlignmentInByte * m_kMemAlignmentInByte;
    }

    bool tryFindFreeOffset(uint64_t sizeInBytes, uint64_t& offsetOut) const
    {
        uint64_t totalSize = m_buffer.getArraySize();

        uint64_t candidateBegin = 0;
        for (const MemSegment& seg : m_usedSegments)
        {
            uint64_t alignedBegin = alignUp(candidateBegin);
            if (alignedBegin + sizeInBytes <= seg.addrBeginIncl)
            {
                offsetOut = alignedBegin;
                return true;
            }
            candidateBegin = seg.addrEndExcl;
        }

        uint64_t alignedBegin = alignUp(candidateBegin);
        if (alignedBegin + sizeInBytes <= totalSize)
        {
            offsetOut = alignedBegin;
            return true;
        }

        return false;
    }
};
