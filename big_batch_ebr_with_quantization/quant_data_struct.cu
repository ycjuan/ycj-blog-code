#include <random>
#include "quant.cuh"

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

void QuantData::initRand(int numDocs, int numReqs, int numInt32)
{
    this->numDocs = numDocs;
    this->numReqs = numReqs;
    this->numInt32 = numInt32;
    this->docMemLayout = docMemLayout;
    this->reqMemLayout = reqMemLayout;

    CHECK_CUDA(cudaMalloc(&d_doc, (size_t)numDocs * numInt32 * sizeof(T_QUANT)));
    CHECK_CUDA(cudaMalloc(&d_req, (size_t)numReqs * numInt32 * sizeof(T_QUANT)));
    CHECK_CUDA(cudaMalloc(&d_rstGpu, (size_t)numDocs * numReqs * sizeof(T_QUANT_RST)));
    CHECK_CUDA(cudaMallocHost(&h_doc, (size_t)numDocs * numInt32 * sizeof(T_QUANT)));
    CHECK_CUDA(cudaMallocHost(&h_req, (size_t)numReqs * numInt32 * sizeof(T_QUANT)));
    CHECK_CUDA(cudaMallocHost(&h_rstCpu, (size_t)numDocs * numReqs * sizeof(T_QUANT_RST)));

    T_QUANT uid = 0;
    for (int i = 0; i < numDocs; i++)
        for (int k = 0; k < numInt32; k++)
            h_doc[getMemAddr(i, k, numDocs, numInt32, docMemLayout)] = uid++;

    uid = 0;
    for (int j = 0; j < numReqs; j++)
    {
        for (int k = 0; k < numInt32; k++)
        {
            size_t addr = getMemAddr(j, k, numReqs, numInt32, reqMemLayout);
            h_req[addr] = uid++;
        }
    }

    CHECK_CUDA(cudaMemcpy(d_doc, h_doc, (size_t)numDocs * numInt32 * sizeof(T_QUANT), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_req, h_req, (size_t)numReqs * numInt32 * sizeof(T_QUANT), cudaMemcpyHostToDevice));
}

void QuantData::free()
{
    cudaFree(d_doc);
    cudaFree(d_req);
    cudaFree(d_rstGpu);
    cudaFreeHost(h_doc);
    cudaFreeHost(h_req);
    cudaFreeHost(h_rstCpu);
}

void QuantData::print()
{
    ostringstream oss;
    oss << "numDocs: " << numDocs << ", numReqs: " << numReqs << ", numInt32: " << numInt32 << ", numBits: " << numInt32 * sizeof(T_QUANT) * 8 << endl;
    oss << "docMemLayout: " << (docMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
    oss << "reqMemLayout: " << (reqMemLayout == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
    oss << "rstLayoutCpu: " << (rstLayoutCpu == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
    oss << "rstLayoutGpu: " << (rstLayoutGpu == ROW_MAJOR ? "ROW_MAJOR" : "COL_MAJOR") << endl;
    cout << oss.str();
}