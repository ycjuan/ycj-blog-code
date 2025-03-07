#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>
#include <omp.h>

#include "util.cuh"
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

ExpData prepareExpData(ExpSetting expSetting)
{
    ExpData expData;
    default_random_engine generator;
    uniform_real_distribution<float> distribution(0.0, 1.0);

    expData.hv_embAll.resize(expSetting.numDocsAll * expSetting.numDims);
    for (int i = 0; i < expSetting.numDocsAll * expSetting.numDims; i++)
        expData.hv_embAll[i] = distribution(generator);

    expData.hv_docIds.resize(expSetting.numDocsAll);
    for (int i = 0; i < expSetting.numDocsAll; i++)
        expData.hv_docIds[i] = i;
    
    ofstream f_bin(expSetting.binaryPath , ios::out | ios::binary);
    if (!f_bin.is_open())
    {
        throw runtime_error("Cannot open file " + expSetting.binaryPath);
    }
    f_bin.write((char *)expData.hv_embAll.data(), expData.hv_embAll.size() * sizeof(float));

    if (expSetting.hasGpu)
    {
        CHECK_CUDA(cudaMalloc(&expData.d_embSelected, expSetting.numDocsSelected * expSetting.numDims * sizeof(float)))
        CHECK_CUDA(cudaMallocHost(&expData.hp_embSelected, expSetting.numDocsSelected * expSetting.numDims * sizeof(float)))
    }
    else
    {
        expData.hp_embSelected = (float *)malloc(expSetting.numDocsSelected * expSetting.numDims * sizeof(float));
    }

    return expData;
}

void runExp(ExpSetting expSetting, ExpData &expData)
{
    double timeMsRandomAcessSum = 0;
    double timeMsH2DSum = 0;
    vector<ifstream> f_bins(expSetting.numThreads);
    for (int i = 0; i < expSetting.numThreads; i++)
    {
        f_bins[i].open(expSetting.binaryPath, ios::in | ios::binary);
        if (!f_bins[i].is_open())
        {
            throw runtime_error("Cannot open file " + expSetting.binaryPath);
        }
    }
    for (int i = 0; i < expSetting.numTrials; i++)
    {
        Timer timer;

        std::shuffle(expData.hv_docIds.begin(), expData.hv_docIds.end(), std::default_random_engine(i));
        cout << "Trial " << i << ": ";
        for (int j = 0; j < 10; j++)
            cout << expData.hv_docIds[j] << " ";
        cout << endl;

        timer.tic();
        omp_set_num_threads(expSetting.numThreads);
        #pragma omp parallel for
        for (int j = 0; j < expSetting.numDocsSelected; j++)
        {
            long offsetSrc = expData.hv_docIds[j] * expSetting.numDims;
            long offsetDst = j * expSetting.numDims;
            if (expSetting.copyMode == MEMCPY)
            {
                memcpy(expData.hp_embSelected + offsetDst, expData.hv_embAll.data() + offsetSrc, expSetting.numDims * sizeof(float));
            }
            else if (expSetting.copyMode == FOR_LOOP)
            {
                for (int k = 0; k < expSetting.numDims; k++)
                {
                    expData.hp_embSelected[offsetDst + k] = expData.hv_embAll[offsetSrc + k];
                }
            }
            else if (expSetting.copyMode == DISK)
            {
                int threadId = omp_get_thread_num();
                auto &f_bin = f_bins[threadId];
                f_bin.seekg(offsetSrc * sizeof(float), ios::beg);
                f_bin.read((char *)(expData.hp_embSelected + offsetDst), expSetting.numDims * sizeof(float));
            }
            else
            {
                throw runtime_error("Unsupported copy mode");
            }
        }
        timeMsRandomAcessSum += timer.tocMs();

        for (int j = 0; j < expSetting.numDocsSelected; j++)
        {
            for (int k = 0; k < expSetting.numDims; k++)
            {
                float a = expData.hp_embSelected[j * expSetting.numDims + k];
                float b = expData.hv_embAll[expData.hv_docIds[j] * expSetting.numDims + k];
                if (a != b)
                {
                    throw runtime_error("Data mismatch");
                }
            }
        }

        timer.tic();
        if (expSetting.hasGpu)
            CHECK_CUDA(cudaMemcpy(expData.d_embSelected, expData.hp_embSelected, expSetting.numDocsSelected * expSetting.numDims * sizeof(float), cudaMemcpyHostToDevice))
        timeMsH2DSum += timer.tocMs();
    }
    //f_bin.close();

    cout << "Average time for random access: " << timeMsRandomAcessSum / expSetting.numTrials << " ms" << endl;
    cout << "Average time for H2D copy: " << timeMsH2DSum / expSetting.numTrials << " ms" << endl;
}

int main()
{
    ExpSetting expSetting;
    expSetting.numTrials = 20;
    expSetting.numDocsAll = 1000000;
    expSetting.numDims = 1024;
    expSetting.numDocsSelected = 100000;
    expSetting.copyMode = DISK;
    expSetting.binaryPath = "bin";
    expSetting.hasGpu = false;
    expSetting.numThreads = 1;

    ExpData expData = prepareExpData(expSetting);

    runExp(expSetting, expData);

    return 0;
}
