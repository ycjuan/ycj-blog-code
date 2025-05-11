#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <future>

#include "util.cuh"

using namespace std;

size_t kNumCopies = 1 << 14; // 16,384
size_t kCopySizeInBytes = 1 << 20; // 1 MiB
int kNumTrials = 10;

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

void runExp(vector<void *> &d_ptrs, vector<void *> &h_ptrs, int numCudaStreams, bool useFuture)
{
    // ----------------
    // Create streams
    vector<cudaStream_t> streams;
    for (int i = 0; i < numCudaStreams; ++i)
    {
        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        streams.push_back(stream);
    }

    // ----------------
    // Run experiment
    Timer timer;
    timer.tic();
    for (int t = 0; t < kNumTrials; ++t)
    {
        if (useFuture)
        {
            vector<future<void>> futures;
            for (int i = 0; i < kNumCopies; ++i)
            {
                auto &stream = streams[i % numCudaStreams];
                futures.push_back(async(launch::async, [=]()
                                        { CHECK_CUDA(cudaMemcpyAsync(d_ptrs[i], h_ptrs[i], kCopySizeInBytes, cudaMemcpyHostToDevice, stream)); }));
            }
            for (int i = 0; i < numCudaStreams; ++i)
            {
                CHECK_CUDA(cudaStreamSynchronize(streams[i]));
            }

            for (auto &f : futures)
            {
                f.get();
            }
        }
        else
        {
            for (int i = 0; i < kNumCopies; ++i)
            {
                auto &stream = streams[i % numCudaStreams];
                CHECK_CUDA(cudaMemcpyAsync(d_ptrs[i], h_ptrs[i], kCopySizeInBytes, cudaMemcpyHostToDevice, stream));
            }
            for (int i = 0; i < numCudaStreams; ++i)
            {
                CHECK_CUDA(cudaStreamSynchronize(streams[i]));
            }
        }
    }
    float elapsedMs = timer.tocMs();
    cout << "numCudaStreams: " << numCudaStreams
         << ", useFuture: " << useFuture
         << ", elapsed time per trial: " << elapsedMs / kNumTrials << " ms" << endl;

    // Free streams
    for (int i = 0; i < numCudaStreams; ++i)
    {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
}

int main()
{
    // ----------------
    // Print config
    cout << "copy size: " << kCopySizeInBytes / (1024 * 1024) << " MB" << endl;
    cout << "num copies: " << kNumCopies << endl;
    cout << "num trials: " << kNumTrials << endl;

    // ----------------
    // Print device info
    printDeviceInfo();

    // ----------------
    // Common variables
    vector<void *> d_ptrs;
    vector<void *> h_ptrs;

    // ----------------
    // Allocate memory
    for (int i = 0; i < kNumCopies; ++i)
    {
        void *d_ptr;
        CHECK_CUDA(cudaMalloc(&d_ptr, kCopySizeInBytes));
        d_ptrs.push_back(d_ptr);

        void *h_ptr;
        CHECK_CUDA(cudaMallocHost(&h_ptr, kCopySizeInBytes));
        h_ptrs.push_back(h_ptr);
    }

    // ----------------
    // Initialize src memory
    for (int i = 0; i < kNumCopies; ++i)
    {
        CHECK_CUDA(cudaMemset(h_ptrs[i], 1, kCopySizeInBytes));
    }

    // ----------------
    // Run experiments
    for (int numCudaStreams = 1; numCudaStreams <= kNumCopies; numCudaStreams *= 2)
    {
        runExp(d_ptrs, h_ptrs, numCudaStreams, false);
        runExp(d_ptrs, h_ptrs, numCudaStreams, true);
    }

    // ----------------
    // Free memory
    for (int i = 0; i < kNumCopies; ++i)
    {
        CHECK_CUDA(cudaFree(d_ptrs[i]));
        CHECK_CUDA(cudaFreeHost(h_ptrs[i]));
    }

    return 0;
}
