#include <string>
#include <stdexcept>
#include <iostream>
#include <random>
#include <sstream>
#include <cublas_v2.h>
#include <type_traits>

#include "util.cuh"
#include "common.cuh"
#include "methods.cuh"

using namespace std;

int kNumDocs = 1 << 19;
int kListSize = 1 << 4;
int kCardinality = 1 << 10;
int kNumTrials = 10;
bool runCpu = true;

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

void checkRst(const vector<Doc> &rstCpu1D, Doc *d_rst, size_t size)
{
    vector<Doc> rstGpu1D(size);
    CHECK_CUDA(cudaMemcpy(rstGpu1D.data(), d_rst, size * sizeof(Doc), cudaMemcpyDeviceToHost));

    if (rstCpu1D.size() != rstGpu1D.size())
    {
        cout << "CPU size: " << rstCpu1D.size() << ", GPU size: " << rstGpu1D.size() << endl;
        throw runtime_error("CPU and GPU sizes do not match!");
    }

    for (int i = 0; i < rstCpu1D.size(); i++)
    {
        const Doc &rstCpu = rstCpu1D[i];
        const Doc &rstGpu = rstGpu1D[i];
        if (rstCpu.docIdx != rstGpu.docIdx || rstCpu.isIn != rstGpu.isIn || rstCpu.docHash != rstGpu.docHash)
        {
            cout << "CPU: " << rstCpu.docIdx << " " << rstCpu.isIn << " " << rstCpu.docHash << endl;
            cout << "GPU: " << rstGpu.docIdx << " " << rstGpu.isIn << " " << rstGpu.docHash << endl;
            throw runtime_error("CPU and GPU results do not match!");
        }
    }

    cout << "\nAll results are correct!" << endl;
}

int main()
{
    Data data;
    data.init(kNumDocs, kListSize, kCardinality);
    data.print();

    Setting setting;
    setting.numTrials = kNumTrials;

    methodGpuNaive(data, setting);

    methodGpuBinarySearch(data, setting);

    if (runCpu)
    {
        methodCpu(data, setting);

        cout << "pass rate = " << data.rstCpu1D.size() * 1.0 / kNumDocs << endl;
        
        cout << "\nChecking results (naive)...";
        checkRst(data.rstCpu1D, data.d_rstGpuNaive, data.rstGpuNaiveSize);

        cout << "\nChecking results (binary search)...";
        checkRst(data.rstCpu1D, data.d_rstGpuBinarySearch, data.rstGpuBinarySearchSize);
    }

    return 0;
}
