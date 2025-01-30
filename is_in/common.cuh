#ifndef COMMON_CUH
#define COMMON_CUH

#include <sstream>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

using namespace std;

struct Setting
{
    int numTrials;
    int numBitTrickLayers;
};

struct Doc
{
    int docIdx;
    int isIn;
    uint64_t docHash;
};

struct Data
{
    int numDocs;
    int listSize;

    vector<Doc> doc1D;
    vector<uint64_t> list1D;
    vector<Doc> rstCpu1D;

    Doc *d_docGpu;
    uint64_t *d_list;
    Doc *d_rstGpuNaive;
    Doc *d_rstGpuBinarySearch;
    Doc *d_rstGpuBitTrick;
    Doc *d_rstGpuBinarySearchPlus;

    int rstGpuNaiveSize;
    int rstGpuBinarySearchSize;
    int rstGpuBitTrickSize;
    int rstGpuBinarySearchPlusSize;

    float timeMsGpuNaive;
    float timeMsGpuBinarySearch;
    float timeMsGpuBitTrick;
    float timeMsGpuBinarySearchPlus;

    void init(int numDocs, int listSize, uint64_t cardinality)
    {
        default_random_engine generator;
        uniform_int_distribution<uint64_t> distribution(0, cardinality - 1);
        uniform_int_distribution<uint64_t> distribution2;

        vector<uint64_t> hashMap(cardinality);
        for (int i = 0; i < cardinality; i++)
        {
            hashMap[i] = distribution2(generator);
        }

        this->numDocs = numDocs;
        this->listSize = listSize;

        doc1D.resize(numDocs);
        list1D.resize(listSize);
        rstCpu1D.resize(numDocs);

        for (int i = 0; i < numDocs; i++)
        {
            doc1D[i].docIdx = i;
            doc1D[i].docHash = hashMap[distribution(generator)];
        }

        for (int i = 0; i < listSize; i++)
        {
            list1D[i] = hashMap[distribution(generator)];
        }
        sort(list1D.begin(), list1D.end());

        cudaMalloc(&d_docGpu, numDocs * sizeof(Doc));
        cudaMallocManaged(&d_list, listSize * sizeof(uint64_t));
        cudaMallocManaged(&d_rstGpuNaive, numDocs * sizeof(Doc));
        cudaMallocManaged(&d_rstGpuBinarySearch, numDocs * sizeof(Doc));
        cudaMallocManaged(&d_rstGpuBitTrick, numDocs * sizeof(Doc));
        cudaMallocManaged(&d_rstGpuBinarySearchPlus, numDocs * sizeof(Doc));

        cudaMemcpy(d_docGpu, doc1D.data(), numDocs * sizeof(Doc), cudaMemcpyHostToDevice);
        cudaMemcpy(d_list, list1D.data(), listSize * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    void free()
    {
        cudaFree(d_docGpu);
        cudaFree(d_list);
        cudaFree(d_rstGpuNaive);
        cudaFree(d_rstGpuBinarySearchPlus);
        cudaFree(d_rstGpuBitTrick);
        cudaFree(d_rstGpuBinarySearch);
    }

    void print()
    {
        ostringstream oss;
        oss << "numDocs: " << numDocs << endl;
        oss << "listSize: " << listSize << endl;
        cout << oss.str();
    }
};

#endif
