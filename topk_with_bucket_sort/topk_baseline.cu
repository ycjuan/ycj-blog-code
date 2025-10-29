#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <algorithm>

#include "topk.cuh"
#include "util.cuh"

std::vector<Doc> retrieveTopkGpuFullSort(Doc *d_doc, int numDocs, int numToRetrieve, float &timeMs)
{
    Timer timer;
    timer.tic();
    thrust::stable_sort(thrust::device, d_doc, d_doc + numDocs, ScorePredicator());
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());

    std::vector<Doc> v_topkDocs(numToRetrieve);
    CHECK_CUDA(cudaMemcpy(v_topkDocs.data(), d_doc, numToRetrieve * sizeof(Doc), cudaMemcpyDeviceToHost));

    timeMs = timer.tocMs();

    return v_topkDocs;
}

std::vector<Doc> retrieveTopkCpuFullSort(std::vector<Doc> &v_doc, int numToRetrieve, float &timeMs)
{
    Timer timer;
    timer.tic();

    stable_sort(v_doc.begin(), v_doc.end(), scoreComparator);
    int numToCopy = std::min(numToRetrieve, (int)v_doc.size());

    std::vector<Doc> v_topkDocs(v_doc.begin(), v_doc.begin() + numToCopy);

    timeMs = timer.tocMs();

    return v_topkDocs;
}
