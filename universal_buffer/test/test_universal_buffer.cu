#include <cassert>
#include <atomic>
#include <chrono>
#include <iostream>
#include <optional>
#include <thread>
#include <vector>
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

    // --------------
    // Test thread safety: 256 threads each call getBuffer and release concurrently.
    // Each thread gets its own 256-byte slice, writes to m_usedBytes must not race.
    {
        std::cout << "======== Test thread safety (256 threads) ========" << std::endl;
        const int      kNumThreads = 256;
        const uint64_t kSliceBytes = 256;
        // Buffer large enough for all threads to hold a slice simultaneously
        UniversalDeviceBuffer buf(kNumThreads * kSliceBytes * 2, "buf");

        std::atomic<int> successCount{0};
        std::atomic<int> errorCount{0};

        auto worker = [&]()
        {
            try
            {
                auto slice = buf.getBuffer(kSliceBytes);
                assert(slice.data() != nullptr);
                assert(reinterpret_cast<uintptr_t>(slice.data()) % 256 == 0);
                ++successCount;
                // slice released here
            }
            catch (const std::exception& e)
            {
                std::cerr << "Thread error: " << e.what() << std::endl;
                ++errorCount;
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(kNumThreads);
        for (int i = 0; i < kNumThreads; ++i)
            threads.emplace_back(worker);
        for (auto& t : threads)
            t.join();

        assert(errorCount == 0);
        assert(successCount == kNumThreads);
        // All slices released — buffer should be fully reclaimed
        assert(buf.getFreeBytes() == buf.getTotalBytes());
    }

    // --------------
    // Test OomPolicy::kThrow (explicit)
    {
        std::cout << "======== Test OomPolicy::kThrow ========" << std::endl;
        UniversalDeviceBuffer buf(512, "buf", OomPolicy::kThrow);
        auto s1 = buf.getBuffer(512);
        bool threw = false;
        try { buf.getBuffer(1); }
        catch (const std::runtime_error&) { threw = true; }
        assert(threw);
    }

    // --------------
    // Test OomPolicy::kWaitSome
    // Main thread fills the buffer, then a background thread releases the slice after a delay.
    // getBuffer on the main thread should block and succeed once the slice is returned.
    {
        std::cout << "======== Test OomPolicy::kWaitSome ========" << std::endl;
        const uint64_t kBufBytes = 1024;
        UniversalDeviceBuffer buf(kBufBytes, "buf", OomPolicy::kWaitSome);

        std::optional<CudaDeviceArray<char>> s1 = buf.getBuffer(kBufBytes); // fills the buffer

        std::thread releaser([&]()
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            s1.reset(); // release s1
        });

        auto s2 = buf.getBuffer(kBufBytes); // should block until releaser runs
        assert(s2.data() != nullptr);
        releaser.join();
    }

    // --------------
    // Test OomPolicy::kWaitAll
    // Main thread fills the buffer, background thread holds a slice then releases.
    // getBuffer with kWaitAll should block until all slices are returned, grow the buffer, then succeed.
    {
        std::cout << "======== Test OomPolicy::kWaitAll ========" << std::endl;
        const uint64_t kBufBytes   = 1024;
        const uint64_t kExtraBytes = 512;
        UniversalDeviceBuffer buf(kBufBytes, "buf", OomPolicy::kWaitAll);

        std::optional<CudaDeviceArray<char>> s1 = buf.getBuffer(kBufBytes); // fills the buffer

        std::thread releaser([&]()
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            s1.reset(); // release s1
        });

        // This should block until s1 is released, then grow buffer and allocate
        auto s2 = buf.getBuffer(kExtraBytes);
        assert(s2.data() != nullptr);
        assert(buf.getTotalBytes() == kBufBytes + kExtraBytes);
        releaser.join();
    }

    std::cout << "All tests passed." << std::endl;
    return 0;
}
