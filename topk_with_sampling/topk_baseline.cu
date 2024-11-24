#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <algorithm>

#include "common.cuh"
#include "topk.cuh"
#include "util.cuh"

using namespace std;

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "[topk_baseline.cu] CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }


vector<Doc> retrieveTopkGpuFullSort(Doc *d_doc, int numDocs, int numToRetrieve, float &timeMs)
{
    CudaTimer timer;
    timer.tic();
    thrust::stable_sort(thrust::device, d_doc, d_doc + numDocs, ScorePredicator());
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());

    vector<Doc> v_topkDocs(numToRetrieve);
    CHECK_CUDA(cudaMemcpy(v_topkDocs.data(), d_doc, numToRetrieve * sizeof(Doc), cudaMemcpyDeviceToHost));

    timeMs = timer.tocMs();

    return v_topkDocs;
}

vector<Doc> retrieveTopkCpuFullSort(vector<Doc> &v_doc, int numToRetrieve, float &timeMs)
{
    CudaTimer timer;
    timer.tic();

    stable_sort(v_doc.begin(), v_doc.end(), scoreComparator);
    int numToCopy = min(numToRetrieve, (int)v_doc.size());

    vector<Doc> v_topkDocs(v_doc.begin(), v_doc.begin() + numToCopy);

    timeMs = timer.tocMs();

    return v_topkDocs;
}
