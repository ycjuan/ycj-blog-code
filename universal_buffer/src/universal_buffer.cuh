#pragma once

#include "cuda_malloc_raii.cuh"
#include <algorithm>
#include <stdexcept>
#include <vector>

struct MemSegment
{
    uint64_t addrBeginIncl; // byte offset from buffer base, inclusive
    uint64_t addrEndExcl;   // byte offset from buffer base, exclusive
};

class UniversalDeviceBuffer
{
public:
    UniversalDeviceBuffer(uint64_t sizeInBytes, std::string name)
        : m_buffer(sizeInBytes, name)
    {
    }

    // Find a free segment of sizeInBytes and return a non-owning CudaDeviceArray<char> wrapping it.
    // Throws if no contiguous free space of the requested size exists.
    CudaDeviceArray<char> getBuffer(uint64_t sizeInBytes)
    {
        uint64_t offset = findFreeOffset(sizeInBytes);
        m_usedSegments.push_back({offset, offset + sizeInBytes});
        std::sort(m_usedSegments.begin(), m_usedSegments.end(),
                  [](const MemSegment& a, const MemSegment& b) { return a.addrBeginIncl < b.addrBeginIncl; });

        char* ptr = m_buffer.data() + offset;
        return CudaDeviceArray<char>(ptr, sizeInBytes, "");
    }

    uint64_t getSize() const { return m_buffer.getArraySize(); }

private:
    // CUDA requires data to be aligned to 256 bytes for optimal memory access.
    // See https://leimao.github.io/blog/CUDA-Data-Alignment/
    static constexpr uint64_t m_kMemAlignmentInByte = 256;

    CudaDeviceArray<char> m_buffer;
    std::vector<MemSegment> m_usedSegments;

    static uint64_t alignUp(uint64_t offset)
    {
        return (offset + m_kMemAlignmentInByte - 1) / m_kMemAlignmentInByte * m_kMemAlignmentInByte;
    }

    uint64_t findFreeOffset(uint64_t sizeInBytes) const
    {
        uint64_t totalSize = m_buffer.getArraySize();

        // Check gap before the first used segment
        uint64_t candidateBegin = 0;
        for (const MemSegment& seg : m_usedSegments)
        {
            uint64_t alignedBegin = alignUp(candidateBegin);
            if (alignedBegin + sizeInBytes <= seg.addrBeginIncl)
                return alignedBegin;
            candidateBegin = seg.addrEndExcl;
        }

        // Check gap after the last used segment (or the whole buffer if empty)
        uint64_t alignedBegin = alignUp(candidateBegin);
        if (alignedBegin + sizeInBytes <= totalSize)
            return alignedBegin;

        throw std::runtime_error("UniversalDeviceBuffer: no free contiguous space of " +
                                 std::to_string(sizeInBytes) + " bytes");
    }
};
