#include <cassert>
#include <cstdint>
#include <math.h>
#include <sstream>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "cabm.cuh"
#include "common.cuh"

// We use uint64_t to store the bit stack, so the max number of elements is 64
constexpr uint32_t g_kMaxBitStackCount = 64;

__device__ void stackPushTrue(uint64_t& bitStack, const uint8_t currBitStackIdx)
{
    uint64_t mask = 1L << currBitStackIdx;
    bitStack = bitStack | mask;
}

__device__ void stackPushFalse(uint64_t& bitStack, const uint8_t currBitStackIdx)
{
    uint64_t mask = ~(1L << currBitStackIdx);
    bitStack = bitStack & mask;
}

__device__ bool stackTop(const uint64_t bitStack, const uint8_t currBitStackIdx)
{
    uint64_t mask = 1L << currBitStackIdx;
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

struct OperandKernelParam
{
    AbmDataGpu reqAbmDataGpu;
    AbmDataGpu docAbmDataGpu;
    CabmOp op;
    uint64_t reqIdx;
    uint64_t numDocs;
    uint64_t* d_bitStacks;
    uint8_t* d_bitStackCounts;
};

__global__ void matchOpKernel(OperandKernelParam param)
{
    uint64_t docIdx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (docIdx < param.numDocs)
    {
        bool rst = matchOp(param.reqAbmDataGpu, param.docAbmDataGpu, param.reqIdx, docIdx, param.op);
        if (rst)
        {
            stackPushTrue(param.d_bitStacks[docIdx], param.d_bitStackCounts[docIdx]);
        }
        else
        {
            stackPushFalse(param.d_bitStacks[docIdx], param.d_bitStackCounts[docIdx]);
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
        uint64_t& bitStack = d_bitStacks[docIdx];
        uint8_t& bitStackCount = d_bitStackCounts[docIdx];
        bool rst1 = stackTop(bitStack, bitStackCount);
        bool rst2 = stackTop(bitStack, bitStackCount);
        bool rst = rst1 & rst2;
        if (rst)
        {
            stackPushTrue(bitStack, bitStackCount);
        }
        else
        {
            stackPushFalse(bitStack, bitStackCount);
        }
    }
}

__global__ void operatorOrKernel(const CabmOp* d_postfixOps,
                                 const uint32_t currOpIdx,
                                 const uint64_t numPostfixOps,
                                 uint64_t* d_bitStacks,
                                 const uint8_t currBitStackIdx,
                                 const uint64_t numDocs)
{
    uint64_t docIdx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (docIdx < numDocs)
    {
        uint64_t& bitStack = d_bitStacks[docIdx];
        bool rst1 = stackTop(bitStack, currBitStackIdx);
        bool rst2 = stackTop(bitStack, currBitStackIdx + 1);
        bool rst = rst1 | rst2;
        if (rst)
        {
            stackPushTrue(bitStack, currBitStackIdx);
        }
        else
        {
            stackPushFalse(bitStack, currBitStackIdx);
        }
    }
}

void cabmGpuOneReq(const AbmDataGpu& reqAbmDataGpu,
                   const AbmDataGpu& docAbmDataGpu,
                   const CabmOp* d_postfixOps,
                   const uint32_t numPostfixOps,
                   uint64_t* d_bitStacks,
                   uint8_t* d_bitStackCounts,
                   const uint64_t numDocs,
                   const uint64_t reqIdx)
{
    const int kBlockSize = 1024;
    const int kGridSize = (numDocs + kBlockSize - 1) / kBlockSize;
    uint8_t currBitStackIdx = 0;
    for (uint32_t opIdx = 0; opIdx < numPostfixOps; opIdx++)
    {
        const CabmOp& op = d_postfixOps[opIdx];
        if (op.isOperand())
        {
            if (currBitStackIdx >= g_kMaxBitStackCount)
            {
                std::ostringstream oss;
                oss << "currBitStackIdx is greater than g_kMaxBitStackCount: " << currBitStackIdx << " >= " << g_kMaxBitStackCount;
                throw std::runtime_error(oss.str());
            }

            OperandKernelParam param;
            param.reqAbmDataGpu = reqAbmDataGpu;
            param.docAbmDataGpu = docAbmDataGpu;
            param.op = op;
            param.reqIdx = reqIdx;
            param.numDocs = numDocs;
            param.d_bitStacks = d_bitStacks;
            param.d_bitStackCounts = d_bitStackCounts;
            if (op.getOpType() == CabmOpType::OPERAND_MATCH)
            {
                matchOpKernel<<<kGridSize, kBlockSize>>>(param);
            }
            else
            {
                assert(false);
            }
            currBitStackIdx++;
        }
        else if (op.isOperator())
        {
            currBitStackIdx -= 2;
            if (op.getOpType() == CabmOpType::OPERATOR_AND)
            {
                operatorAndKernel<<<kGridSize, kBlockSize>>>(d_postfixOps, opIdx, numPostfixOps, d_bitStacks, d_bitStackCounts, numDocs);
            }
            else if (op.getOpType() == CabmOpType::OPERATOR_OR)
            {
                operatorOrKernel<<<kGridSize, kBlockSize>>>(d_postfixOps, opIdx, numPostfixOps, d_bitStacks, currBitStackIdx, numDocs);
            }
            else
            {
                assert(false);
            }    
        }
        CHECK_CUDA(cudaDeviceSynchronize())
        CHECK_CUDA(cudaGetLastError())
    }
}

/*
void cabmGpuOneReq(const AbmDataGpu& reqAbmDataGpu,
                   const AbmDataGpu& docAbmDataGpu,
                   const CabmOp* d_postfixOps,
                   const uint32_t numPostfixOps,
                   uint64_t* d_bitStacks,
                   uint8_t* d_bitStackCounts,
                   const uint64_t numDocs,
                   const uint64_t reqIdx)
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