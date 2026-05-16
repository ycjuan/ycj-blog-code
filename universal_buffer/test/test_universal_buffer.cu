#include <cassert>
#include <iostream>
#include "universal_buffer.cuh"

int main()
{
    const uint64_t kTotalBytes = 1024 * 1024; // 1 MB

    // --------------
    // Test getTotalBytes / getFreeBytes initial state
    {
        std::cout << "======== Test getTotalBytes / getFreeBytes ========" << std::endl;
        UniversalDeviceBuffer buf(kTotalBytes, "buf");
        assert(buf.getTotalBytes() == kTotalBytes);
        assert(buf.getFreeBytes() == kTotalBytes);
    }

    // --------------
    // Test getBuffer reduces free bytes
    {
        std::cout << "======== Test getBuffer reduces free bytes ========" << std::endl;
        UniversalDeviceBuffer buf(kTotalBytes, "buf");
        const uint64_t kAllocBytes = 1024;
        {
            auto slice = buf.getBuffer(kAllocBytes);
            assert(slice.data() != nullptr);
            assert(slice.getArraySize() == kAllocBytes);
            assert(buf.getFreeBytes() == kTotalBytes - kAllocBytes);
        }
        // slice out of scope — next getBuffer call should prune and restore free bytes
        auto slice2 = buf.getBuffer(kAllocBytes);
        assert(buf.getFreeBytes() == kTotalBytes - kAllocBytes);
    }

    // --------------
    // Test multiple allocations and auto-release
    {
        std::cout << "======== Test multiple allocations and auto-release ========" << std::endl;
        UniversalDeviceBuffer buf(kTotalBytes, "buf");
        const uint64_t kAllocBytes = 256;
        {
            auto s1 = buf.getBuffer(kAllocBytes);
            auto s2 = buf.getBuffer(kAllocBytes);
            assert(buf.getFreeBytes() == kTotalBytes - 2 * kAllocBytes);
            assert(s1.data() != s2.data());
        }
        // Both released — next alloc should reclaim all space
        auto s3 = buf.getBuffer(kAllocBytes);
        assert(buf.getFreeBytes() == kTotalBytes - kAllocBytes);
    }

    // --------------
    // Test alignment: returned pointer should be 256-byte aligned
    {
        std::cout << "======== Test alignment ========" << std::endl;
        UniversalDeviceBuffer buf(kTotalBytes, "buf");
        auto s1 = buf.getBuffer(100); // non-aligned size
        auto s2 = buf.getBuffer(100);
        assert(reinterpret_cast<uintptr_t>(s2.data()) % 256 == 0);
    }

    // --------------
    // Test out-of-memory throws
    {
        std::cout << "======== Test out-of-memory throws ========" << std::endl;
        UniversalDeviceBuffer buf(512, "buf");
        auto s1 = buf.getBuffer(512);
        bool threw = false;
        try { buf.getBuffer(1); }
        catch (const std::runtime_error&) { threw = true; }
        assert(threw);
    }

    std::cout << "All tests passed." << std::endl;
    return 0;
}
