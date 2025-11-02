#pragma once

#include <vector>
#include <algorithm>
#include <thrust/sort.h>


#include "util.cuh"

template <typename T, class ScorePredicator>
std::vector<T> retrieveTopkGpuFullSort(T* d_doc, int numDocs, int numToRetrieve)
{
    thrust::stable_sort(thrust::device, d_doc, d_doc + numDocs, ScorePredicator());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    std::vector<T> v_topkDocs(numToRetrieve);
    CHECK_CUDA(cudaMemcpy(v_topkDocs.data(), d_doc, numToRetrieve * sizeof(T), cudaMemcpyDeviceToHost));

    return v_topkDocs;
}

template <typename T, class ScorePredicator>
std::vector<T> retrieveTopkCpuFullSort(std::vector<T>& v_doc, int numToRetrieve)
{
    stable_sort(v_doc.begin(), v_doc.end(), ScorePredicator());
    int numToCopy = std::min(numToRetrieve, (int)v_doc.size());

    std::vector<T> v_topkDocs(v_doc.begin(), v_doc.begin() + numToCopy);

    return v_topkDocs;
}