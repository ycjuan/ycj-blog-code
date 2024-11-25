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

void retrieveTopkCpu(TopkParam &param)
{
    CudaTimer timer;
    timer.tic();

    for (int j = 0; j < param.numReqs; j++)
    {
        if (j % 8 == 0)
        {
            cout << "retrieving topk for req " << j << endl;
        }
        vector<Pair> v_doc;
        for (int i = 0; i < param.numDocs; i++)
        {
            Pair doc;
            doc.reqIdx = j;
            doc.docIdx = i;
            doc.score = param.dm_score[j * param.numDocs + i];
            v_doc.push_back(doc);
        }

        stable_sort(v_doc.begin(), v_doc.end(), scoreComparator);
        for (int i = 0; i < param.numToRetrieve; i++)
        {
            param.hp_rstCpu[j * param.numToRetrieve + i] = v_doc[i];
        }
    }

    param.cpuTimeMs = timer.tocMs();
}
