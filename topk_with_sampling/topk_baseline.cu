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

void retrieveTopkCpuFullSort(float *dm_score, size_t numReqs, size_t numDocs, size_t numToRetrieve, Pair *dm_rst, float &timeMs)
{
    CudaTimer timer;
    timer.tic();

    for (int j = 0; j < numReqs; j++)
    {
        vector<Pair> v_doc;
        for (int i = 0; i < numDocs; i++)
        {
            Pair doc;
            doc.reqId = j;
            doc.docId = i;
            doc.score = dm_score[j * numDocs + i];
            v_doc.push_back(doc);
        }

        sort(v_doc.begin(), v_doc.end(), scoreComparator);
        for (int i = 0; i < numToRetrieve; i++)
        {
            dm_rst[j * numToRetrieve + i] = v_doc[i];
        }
    }

    timeMs = timer.tocMs();
}
