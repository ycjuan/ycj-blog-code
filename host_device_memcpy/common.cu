#include <string>
#include <sstream>
#include <random>
#include <iostream>
#include "common.cuh"

using namespace std;

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

void Data::malloc(size_t numRows, size_t numCols)
{
    // -----------------------
    // meta data
    this->numRows = numRows;
    this->numCols = numCols;
    this->dataSize = numRows * numCols;
    dataSizeInBytes = numRows * numCols * sizeof(T);

    // -----------------------
    // allocate memory
    h_data_src = (T *)std::malloc(dataSizeInBytes);
    h_data_dst = (T *)std::malloc(dataSizeInBytes);
    CHECK_CUDA(cudaMallocHost(&hp_data_src, dataSizeInBytes));
    CHECK_CUDA(cudaMallocHost(&hp_data_dst, dataSizeInBytes));
    CHECK_CUDA(cudaMalloc(&d_data_src, dataSizeInBytes));
    CHECK_CUDA(cudaMalloc(&d_data_dst, dataSizeInBytes));

    // -----------------------
    // assign random data
    default_random_engine generator;
    uniform_real_distribution<T> distribution(0.0, 1.0);
    for (size_t i = 0; i < dataSize; i++)
    {
        h_data_src[i] = distribution(generator);
    }

    // -----------------------
    // copy data to other pointers
    CHECK_CUDA(cudaMemcpy(hp_data_src, h_data_src, dataSizeInBytes, cudaMemcpyHostToHost));
    CHECK_CUDA(cudaMemcpy(d_data_src, h_data_src, dataSizeInBytes, cudaMemcpyHostToDevice));
}

void Data::print()
{
    cout << "numRows: " << numRows << endl;
    cout << "numCols: " << numCols << endl;
    cout << "dataSize: " << dataSize << endl;
    cout << "dataSizeInGiB: " << dataSizeInBytes / 1024.0 / 1024.0 / 1024.0 << " GiB" << endl;
}