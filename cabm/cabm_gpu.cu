#include <vector>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <math.h>
#include <cassert>
#include <numeric>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#ifndef CABM_H
#define CABM_H
#include "cabm.cuh"
#endif

#define CHECK_CUDA(func)                                 \
{                                                        \
    cudaError_t status = (func);                         \
    if (status != cudaSuccess) {                         \
        std::string error = "CUDA API failed at line "   \
            + std::to_string(__LINE__) + " with error: " \
            + cudaGetErrorString(status) + "\n";         \
        throw std::runtime_error(error);                 \
    }                                                    \
}

using namespace std;
    
const int BLOCK_SIZE = 1024;

__device__ long getMemAddrRowMajorDevice(int row, int col, int numRows, int numCols) {
    return (long)row * numCols + col;
}

__device__ void stackPushTrue(uint64_t &bs, int &bsCount) {
    uint64_t mask = 1L << bsCount;
    bs = bs | mask;
    bsCount++;
}

__device__ void stackPushFalse(uint64_t &bs, int &bsCount) {
    uint64_t mask = ~(1L << bsCount);
    bs = bs & mask;
    bsCount++;
}

__device__ bool stackPop(uint64_t &bs, int &bsCount) {
    bsCount--;
    uint64_t mask = 1L << bsCount;
    uint64_t tmp = bs & mask;
    return tmp > 0L;
}

__device__ bool evaluateSingleOp(const CabmGpuParam &param, int i, const CabmOp &op) {
    int docOffsetBegin = param.d_docTbrOffsets[getMemAddrRowMajorDevice(i, op.clause, param.numDocs, param.numClauses+1)]; 
    int docOffsetEnd = param.d_docTbrOffsets[getMemAddrRowMajorDevice(i, op.clause+1, param.numDocs, param.numClauses+1)]; 
    bool rst = false;
    for (int j = docOffsetBegin; j < docOffsetEnd; j++) {
        long doc_tbr = param.d_docTbrAttr[getMemAddrRowMajorDevice(i, j, param.numDocs, param.docMaxNumTbrAttr)];
        if (doc_tbr == op.attr) {
            rst = true;
            break;
        }
    }
    if (op.type == CABM_OP_TYPE_ATTR_NEGATION)
        rst = !rst;
    return rst;
}

__global__ void cabmKernel(CabmGpuParam param) {

    long m = (long)blockIdx.x*blockDim.x+threadIdx.x + param.offsetA;

    if (m < param.msgSize) {
        int i;
        int r;
        Msg msg;

        if (param.d_msgInit != nullptr) {
            i = m / param.numReqs;
            r = m % param.numReqs;
            msg = param.d_msgInit[i];
            msg.i = i;
            msg.r = r;
        } else {
            msg = param.d_msg[m];
            i = msg.i;
            r = msg.r;
        }

        bool finalRst;
        if (msg.score != 0) {
            uint64_t bs; 
            int bsCount = 0; // bsCount-1 indicate the current head
            // bs stands for "bitStack". since we only need to store the "binary result" of each operand, we only need 1 bit for each operand.
            // here we use a 64-bit integer, so it can hold up to 64 elements in the stack
            // "bsCount" is used to indicate how many element there are in the bit stack

            for (int c = 0; c < param.reqPostfixExprLength; c++) {
                CabmOp op = param.d_reqPostfixOp[c];

                if (op.isOperand) {
                    bool rst = evaluateSingleOp(param, i, op);

                    if (rst)
                        stackPushTrue(bs, bsCount);
                    else
                        stackPushFalse(bs, bsCount);

                } else {

                    bool rst1 = stackPop(bs, bsCount);
                    bool rst2 = stackPop(bs, bsCount);
                    
                    bool rst;
                    if (op.type == CABM_OP_TYPE_AND)
                        rst = rst1 & rst2;
                    else if (op.type == CABM_OP_TYPE_OR)
                        rst = rst1 | rst2;
                    else
                        assert(false);

                    if (rst)
                        stackPushTrue(bs, bsCount);
                    else
                        stackPushFalse(bs, bsCount);

                } // end if (op.isOperand)

            } // end for loop

            finalRst = stackPop(bs, bsCount);

        } else {

            finalRst = false;

        }

        msg.score = finalRst? 1.0 : 0.0;
        param.d_msgBuffer[m] = msg;
    }
}

struct nonZeroPredicator {
    __host__ __device__ bool operator()(const Msg x) {
        return x.score != 0;
    }
};

struct RowPredicator {
  __host__ __device__ bool operator()(const Msg& a, const Msg& b) {
      return a.i < b.i;
  }
};

struct ScorePredicator {
  __host__ __device__ bool operator()(const Msg& a, const Msg& b) {
      return a.score > b.score;
  }
};

void cabmGpu(CabmGpuParam &param) {

    // Execute the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);    
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (param.k == 0) {
        param.offsetA = 0;
        int GRID_SIZE = max(1, (int)ceil((double)param.msgSize/BLOCK_SIZE));
        cabmKernel<<<GRID_SIZE, BLOCK_SIZE>>>(param);
        CHECK_CUDA(cudaDeviceSynchronize())
        Msg* d_endPtr = thrust::copy_if(thrust::device, param.d_msgBuffer, param.d_msgBuffer + param.msgSize, param.d_msg, nonZeroPredicator());
        param.msgSize = d_endPtr - param.d_msg;
    } else {
        int iterSize = max(16384, param.k);
        int GRID_SIZE = max(1, (int)ceil((double)iterSize/BLOCK_SIZE));
        param.offsetA = 0;
        param.offsetB = 0;
        while (true) {
            //Msg *d_msgBegin = param.d_msg + param.offsetA;
            //Msg *d_msgEnd   = param.d_msg + min(param.msgSize, param.offsetA + iterSize);
            //thrust::stable_sort(thrust::device, d_msgBegin, d_msgEnd, RowPredicator());
            cabmKernel<<<GRID_SIZE, BLOCK_SIZE>>>(param);
            CHECK_CUDA(cudaDeviceSynchronize())
            Msg *d_msgBufferBegin = param.d_msgBuffer + param.offsetA;
            Msg *d_msgBufferEnd   = param.d_msgBuffer + min(param.msgSize, param.offsetA + iterSize);
            Msg *d_msg            = param.d_msg + param.offsetB;
            Msg* d_endPtr = thrust::copy_if(thrust::device, d_msgBufferBegin, d_msgBufferEnd, d_msg, nonZeroPredicator());
            param.offsetA += iterSize;
            param.offsetB += d_endPtr - d_msg;
            if (param.offsetA >= param.msgSize || param.offsetB >= param.k)
                break;
        }
        param.msgSize = param.offsetB;
        //thrust::stable_sort(thrust::device, param.d_msg, param.d_msg + param.msgSize, ScorePredicator());
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&(param.timeMs), start, stop);
    cudaEventDestroy(start);    
    cudaEventDestroy(stop);
}
