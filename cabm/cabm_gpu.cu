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
    int reqOffsetIter = reqAbmDataGpu.getOffset_d(reqIdx, op.getReqFieldIdx());
    int docOffsetIter = docAbmDataGpu.getOffset_d(docIdx, op.getDocFieldIdx());
    int reqOffsetEnd = reqAbmDataGpu.getOffset_d(reqIdx, op.getReqFieldIdx() + 1);
    int docOffsetEnd = docAbmDataGpu.getOffset_d(docIdx, op.getDocFieldIdx() + 1);

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

struct OperatorKernelParam
{
    CabmOp op;
    uint32_t currOpIdx;
    uint64_t numPostfixOps;
    uint64_t* d_bitStacks;
    uint8_t* d_bitStackCounts;
    uint64_t numDocs;
};

__global__ void operatorKernel(OperatorKernelParam param)
{
    uint64_t docIdx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (docIdx < param.numDocs)
    {
        uint64_t& bitStack = param.d_bitStacks[docIdx];
        uint8_t& bitStackCount = param.d_bitStackCounts[docIdx];
        bool rst1 = stackTop(bitStack, bitStackCount);
        bool rst2 = stackTop(bitStack, bitStackCount);
        bool rst = (param.op.getOpType() == CabmOpType::OPERATOR_AND) ? (rst1 & rst2) : (rst1 | rst2);
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

struct CabmGpuParam
{
    AbmDataGpu reqAbmDataGpu;
    AbmDataGpu docAbmDataGpu;
    CabmOp* d_postfixOps;
    uint32_t numPostfixOps;
    uint64_t* d_bitStacks;
    uint8_t* d_bitStackCounts;
    uint64_t numDocs;
    uint64_t numReqs;
};

void cabmGpu(CabmGpuParam param)
{
    const int kBlockSize = 1024;
    const int kGridSize = (param.numDocs + kBlockSize - 1) / kBlockSize;

    for (uint32_t reqIdx = 0; reqIdx < param.numReqs; reqIdx++)
    {
        uint8_t currBitStackIdx = 0;
        for (uint32_t opIdx = 0; opIdx < param.numPostfixOps; opIdx++)
        {
            const CabmOp& op = param.d_postfixOps[opIdx];
            if (op.isOperand())
            {
                if (currBitStackIdx >= g_kMaxBitStackCount)
                {
                    std::ostringstream oss;
                    oss << "currBitStackIdx is greater than g_kMaxBitStackCount: " << currBitStackIdx
                        << " >= " << g_kMaxBitStackCount;
                    throw std::runtime_error(oss.str());
                }

                OperandKernelParam param;
                param.reqAbmDataGpu = param.reqAbmDataGpu;
                param.docAbmDataGpu = param.docAbmDataGpu;
                param.op = op;
                param.reqIdx = reqIdx;
                param.numDocs = param.numDocs;
                param.d_bitStacks = param.d_bitStacks;
                param.d_bitStackCounts = param.d_bitStackCounts;
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

                OperatorKernelParam param;
                param.op = op;
                param.currOpIdx = opIdx;
                param.numPostfixOps = param.numPostfixOps;
                param.d_bitStacks = param.d_bitStacks;
                param.d_bitStackCounts = param.d_bitStackCounts;
                param.numDocs = param.numDocs;
                operatorKernel<<<kGridSize, kBlockSize>>>(param);
            }
            CHECK_CUDA(cudaDeviceSynchronize())
            CHECK_CUDA(cudaGetLastError())
        }
    }
}