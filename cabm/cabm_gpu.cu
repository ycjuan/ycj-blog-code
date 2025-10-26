#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <math.h>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <vector>

#include "cabm.cuh"

constexpr uint32_t g_kMaxBitStackCount = 64; // We use uint64_t to store the bit stack, so the max number of elements is 64

__device__ void stackPushTrue(uint64_t& bitStack, uint8_t& bitStackCount)
{
    uint64_t mask = 1L << bitStackCount;
    bitStack = bitStack | mask;
    bitStackCount++;
}

__device__ void stackPushFalse(uint64_t& bitStack, uint8_t& bitStackCount)
{
    uint64_t mask = ~(1L << bitStackCount);
    bitStack = bitStack & mask;
    bitStackCount++;
}

__device__ bool stackPop(uint64_t& bitStack, uint8_t& bitStackCount)
{
    bitStackCount--;
    uint64_t mask = 1L << bitStackCount;
    uint64_t tmp = bitStack & mask;
    return tmp > 0L;
}

__device__ bool matchOp(const AbmDataGpu& reqAbmDataGpu,
                        const AbmDataGpu& docAbmDataGpu,
                        const int reqIdx,
                        const int docIdx,
                        const CabmOp& op)
{
    int reqOffsetIter = reqAbmDataGpu.getOffset_d(reqIdx, op.getReqFieldIdx_dh());
    int docOffsetIter = docAbmDataGpu.getOffset_d(docIdx, op.getDocFieldIdx_dh());
    int reqOffsetEnd = reqAbmDataGpu.getOffset_d(reqIdx, op.getReqFieldIdx_dh() + 1);
    int docOffsetEnd = docAbmDataGpu.getOffset_d(docIdx, op.getDocFieldIdx_dh() + 1);

    while (reqOffsetIter < reqOffsetEnd && docOffsetIter < docOffsetEnd)
    {
        long reqVal = reqAbmDataGpu.getVal_d(reqIdx, reqOffsetIter);
        long docVal = docAbmDataGpu.getVal_d(docIdx, docOffsetIter);
        if (reqVal == docVal)
        {
            return true;
        }
        else if (reqVal < docVal)
        {
            reqOffsetIter++;
        }
        else
        {
            docOffsetIter++;
        }
    }

    return false;
}

__global__ void matchOpKernel(const AbmDataGpu& reqAbmDataGpu,
                              const AbmDataGpu& docAbmDataGpu,
                              const CabmOp& op,
                              const ReqDocPair *d_reqDocPairs,    
                              const uint64_t numReqDocPairs,
                              uint64_t *d_bitStacks,
                              uint8_t *d_bitStackCounts,
                              const uint8_t maxBitStackCount)
{
    uint64_t reqDocPairIdx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (reqDocPairIdx < numReqDocPairs)
    {
        const ReqDocPair &reqDocPair = d_reqDocPairs[reqDocPairIdx];
        bool rst = matchOp(reqAbmDataGpu, docAbmDataGpu, reqDocPair.reqIdx, reqDocPair.docIdx, op);
        if (rst)
        {
            stackPushTrue(d_bitStacks[reqDocPairIdx], d_bitStackCounts[reqDocPairIdx]);
        }
        else
        {
            stackPushFalse(d_bitStacks[reqDocPairIdx], d_bitStackCounts[reqDocPairIdx]);
        }
    }
}

__global__ void operatorAndKernel(const CabmOp* d_postfixOps,
                                  const uint32_t currOpIdx,
                                  const uint64_t numPostfixOps,
                                  uint64_t* d_bitStacks,
                                  uint8_t* d_bitStackCounts,
                                  const uint64_t numDocs)
{
    uint64_t docIdx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (docIdx < numDocs)
    {
        uint64_t &bitStack = d_bitStacks[docIdx];
        uint8_t &bitStackCount = d_bitStackCounts[docIdx];
        bool rst1 = stackPop(bitStack, bitStackCount);
        bool rst2 = stackPop(bitStack, bitStackCount);
        bool rst = rst1 & rst2;
        stackPushTrue(bitStack, bitStackCount);
    }
}

__global__ void operatorOrKernel(const CabmOp* d_postfixOps,
                                  const uint32_t currOpIdx,
                                  const uint64_t numPostfixOps,
                                  uint64_t* d_bitStacks,
                                  uint8_t* d_bitStackCounts,
                                  const uint64_t numDocs)
{
    uint64_t docIdx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (docIdx < numDocs)
    {
        uint64_t &bitStack = d_bitStacks[docIdx];
        uint8_t &bitStackCount = d_bitStackCounts[docIdx];
        bool rst1 = stackPop(bitStack, bitStackCount);
        bool rst2 = stackPop(bitStack, bitStackCount);
        bool rst = rst1 | rst2;
        stackPushTrue(bitStack, bitStackCount);
    }
}

/*

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
            // bs stands for "bitStack". since we only need to store the "binary result" of each operand, we only need 1
bit for each operand.
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
        Msg* d_endPtr = thrust::copy_if(thrust::device, param.d_msgBuffer, param.d_msgBuffer + param.msgSize,
param.d_msg, nonZeroPredicator()); param.msgSize = d_endPtr - param.d_msg; } else { int iterSize = max(16384, param.k);
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
            Msg* d_endPtr = thrust::copy_if(thrust::device, d_msgBufferBegin, d_msgBufferEnd, d_msg,
nonZeroPredicator()); param.offsetA += iterSize; param.offsetB += d_endPtr - d_msg; if (param.offsetA >= param.msgSize
|| param.offsetB >= param.k) break;
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
*/