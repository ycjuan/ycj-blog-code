#ifndef DATA_STRUCT_CUH
#define DATA_STRUCT_CUH

#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace std;

struct ItemCpu
{
    int uid;
    int centroidId;
    vector<float> emb;
    float randAttr;
    float bid;
};

struct CentroidCpu
{
    int centroidId;
    vector<float> emb;
};

struct ItemGpu
{
    int uid;
    int centroidId;
    float randAttr;
    float bid;
};

struct ItemDataGpu
{
    ItemGpu *d_item;
    size_t numItems;
    size_t size_d_item;
    size_t size_byte_d_item;

    float *d_emb;
    size_t embDim;
    size_t size_d_emb;
    size_t size_byte_d_emb;

    void malloc(size_t numItems, size_t embDim)
    {
        this->numItems = numItems;
        size_d_item = numItems;
        size_byte_d_item = numItems * sizeof(ItemGpu);
        cudaError_t cudaError = cudaMallocManaged(&d_item, size_byte_d_item);
        if (cudaError != cudaSuccess)
        {
            throw runtime_error("cudaMallocManaged failed: " + string(cudaGetErrorString(cudaError)));
        }

        this->embDim = embDim;
        size_d_emb = numItems * embDim;
        size_byte_d_emb = size_d_emb * sizeof(float);
        cudaError = cudaMallocManaged(&d_emb, size_byte_d_emb);
        if (cudaError != cudaSuccess)
        {
            throw runtime_error("cudaMallocManaged failed: " + string(cudaGetErrorString(cudaError)));
        }
    }

    void init(const vector<ItemCpu> &itemCpu1D)
    {
        malloc(itemCpu1D.size(), itemCpu1D[0].emb.size());
        for (int i = 0; i < numItems; i++)
        {
            d_item[i].uid = itemCpu1D[i].uid;
            d_item[i].centroidId = itemCpu1D[i].centroidId;
            d_item[i].randAttr = itemCpu1D[i].randAttr;
            d_item[i].bid = itemCpu1D[i].bid;
            for (int j = 0; j < embDim; j++)
            {
                d_emb[getEmbMemAddr(i, j)] = itemCpu1D[i].emb[j];
            }
        }
    }

    void reset()
    {
        if (d_item != nullptr)
        {
            cudaFree(d_item);
        }
        if (d_emb != nullptr)
        {
            cudaFree(d_emb);
        }
        size_d_item = 0;
        size_byte_d_item = 0;
    }

    __device__ __host__ size_t getEmbMemAddr(int i, int j) const
    {
        return (size_t)j * numItems + i;
    }

    __device__ __host__ float getEmb(int i, int j) const
    {
        return d_emb[getEmbMemAddr(i, j)];
    }
};

struct ReqDocPair
{
    int reqIdx;
    int reqCentroidId;
    int docIdx;
    int docCentroidId;
    float score;
};

struct ReqDocPairDataGpu
{
    ReqDocPair *d_data;
    size_t size_d_data;
    size_t size_byte_d_data;

    void malloc(size_t size_d_data)
    {
        this->size_d_data = size_d_data;
        this->size_byte_d_data = size_d_data * sizeof(ReqDocPair);
        cudaError_t cudaError = cudaMallocManaged(&d_data, size_byte_d_data);
        if (cudaError != cudaSuccess)
        {
            throw runtime_error("cudaMallocManaged failed: " + string(cudaGetErrorString(cudaError)));
        }
    }

    void init(const ItemCpu &req, const vector<ItemCpu> &docs)
    {
        malloc(docs.size());
        for (int docIdx = 0; docIdx < docs.size(); docIdx++)
        {
            ReqDocPair &pair = d_data[docIdx];
            pair.reqIdx = req.uid;
            pair.reqCentroidId = req.centroidId;
            pair.docIdx = docs[docIdx].uid;
            pair.docCentroidId = docs[docIdx].centroidId;
            pair.score = 0.0f;
        }
    }

    void reset()
    {
        if (d_data != nullptr)
        {
            cudaFree(d_data);
        }
        size_d_data = 0;
        size_byte_d_data = 0;
    }
};

#endif // DATA_STRUCT_CUH