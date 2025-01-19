#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <algorithm>
#include <omp.h>

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

    omp_set_num_threads(8);
    #pragma omp parallel for
    for (int reqIdx = 0; reqIdx < param.numReqs; reqIdx++)
    {
        vector<Pair> v_doc;
        for (int docIdx = 0; docIdx < param.numDocs; docIdx++)
        {
            Pair doc;
            doc.reqIdx = reqIdx;
            doc.docIdx = docIdx;
            doc.score = param.h_score[getMemAddr(reqIdx, docIdx, param.numDocs)];
            v_doc.push_back(doc);
        }

        stable_sort(v_doc.begin(), v_doc.end(), scoreComparator);
        for (int docIdx = 0; docIdx < param.numToRetrieve; docIdx++)
        {
            param.h_rstCpu[getMemAddr(reqIdx, docIdx, param.numToRetrieve)] = v_doc[docIdx];
        }

        if (reqIdx % 8 == 0)
        {
            cout << "retrieved topk for req " << reqIdx << endl;
        }
    }

    param.cpuTimeMs = timer.tocMs();
}
