#include <cassert>
#include <cstdint>
#include <math.h>
#include <ostream>
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

__global__ void copyRstKernel(uint8_t* d_rst, uint64_t* d_bitStacks, uint64_t numDocs)
{
    uint64_t docIdx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (docIdx < numDocs)
    {
        d_rst[docIdx] = stackTop(d_bitStacks[docIdx], 0);
    }
}

struct CabmGpuParam
{
    AbmDataGpu reqAbmDataGpu;
    AbmDataGpu docAbmDataGpu;
    std::vector<CabmOp> postfixOps;
    uint64_t* d_bitStacks;
    uint8_t* d_bitStackCounts;
    uint64_t numDocs;
    uint64_t numReqs;
    uint8_t* d_rst;
};

void cabmGpu(CabmGpuParam param)
{
    const int kBlockSize = 1024;
    const int kGridSize = (param.numDocs + kBlockSize - 1) / kBlockSize;

    for (uint32_t reqIdx = 0; reqIdx < param.numReqs; reqIdx++)
    {
        uint8_t currBitStackIdx = 0;
        CHECK_CUDA(cudaMemset(param.d_bitStacks, 0, param.numDocs * sizeof(uint64_t)));
        CHECK_CUDA(cudaMemset(param.d_bitStackCounts, 0, param.numDocs * sizeof(uint8_t)));
        for (const auto& op : param.postfixOps)
        {
            if (op.isOperand())
            {
                if (currBitStackIdx >= g_kMaxBitStackCount)
                {
                    std::ostringstream oss;
                    oss << "currBitStackIdx is greater than g_kMaxBitStackCount: " << currBitStackIdx
                        << " >= " << g_kMaxBitStackCount;
                    throw std::runtime_error(oss.str());
                }

                OperandKernelParam operandKernelParam;
                operandKernelParam.reqAbmDataGpu = param.reqAbmDataGpu;
                operandKernelParam.docAbmDataGpu = param.docAbmDataGpu;
                operandKernelParam.op = op;
                operandKernelParam.reqIdx = reqIdx;
                operandKernelParam.numDocs = param.numDocs;
                operandKernelParam.d_bitStacks = param.d_bitStacks;
                operandKernelParam.d_bitStackCounts = param.d_bitStackCounts;
                if (op.getOpType() == CabmOpType::OPERAND_MATCH)
                {
                    matchOpKernel<<<kGridSize, kBlockSize>>>(operandKernelParam);
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

                OperatorKernelParam operatorKernelParam;
                operatorKernelParam.op = op;
                operatorKernelParam.d_bitStacks = param.d_bitStacks;
                operatorKernelParam.d_bitStackCounts = param.d_bitStackCounts;
                operatorKernelParam.numDocs = param.numDocs;
                operatorKernel<<<kGridSize, kBlockSize>>>(operatorKernelParam);
            }
            CHECK_CUDA(cudaDeviceSynchronize())
            CHECK_CUDA(cudaGetLastError())
        }

        if (currBitStackIdx != 0)
        {
            std::ostringstream oss;
            oss << "currBitStackIdx is not 0: " << currBitStackIdx;
            throw std::runtime_error(oss.str());
        }

        copyRstKernel<<<kGridSize, kBlockSize>>>(param.d_rst, param.d_bitStacks, param.numDocs);
        CHECK_CUDA(cudaDeviceSynchronize())
        CHECK_CUDA(cudaGetLastError())
    }
}

bool evaluatePostfixGpuWrapped(std::vector<CabmOp> postfix1D,
                               const std::vector<std::vector<long>>& reqData2D,
                               const std::vector<std::vector<long>>& docData2D)
{
    AbmDataGpu reqAbmDataGpu;
    AbmDataGpu docAbmDataGpu;
    reqAbmDataGpu.init({reqData2D}, true);
    docAbmDataGpu.init({docData2D}, true);

    uint8_t* d_rst;
    CHECK_CUDA(cudaMallocManaged(&d_rst, 1 * sizeof(uint8_t)));
    uint64_t* d_bitStacks;
    CHECK_CUDA(cudaMallocManaged(&d_bitStacks, 1 * sizeof(uint64_t)));
    uint8_t* d_bitStackCounts;
    CHECK_CUDA(cudaMallocManaged(&d_bitStackCounts, 1 * sizeof(uint8_t)));

    CabmGpuParam param;
    param.d_rst = d_rst;
    param.d_bitStacks = d_bitStacks;
    param.d_bitStackCounts = d_bitStackCounts;
    param.numDocs = 1;
    param.numReqs = 1;
    param.postfixOps = postfix1D;
    param.reqAbmDataGpu = reqAbmDataGpu;
    param.docAbmDataGpu = docAbmDataGpu;

    cabmGpu(param);

    uint8_t rst = d_rst[0];

    CHECK_CUDA(cudaFree(d_rst));
    CHECK_CUDA(cudaFree(d_bitStacks));
    CHECK_CUDA(cudaFree(d_bitStackCounts));

    return rst;
}