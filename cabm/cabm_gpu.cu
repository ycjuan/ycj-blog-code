#include <cstdint>
#include <math.h>
#include <ostream>
#include <sstream>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "cabm.cuh"
#include "macro.cuh"
#include "util.cuh"

// We use uint64_t to store the bit stack, so the max number of elements is 64
constexpr uint32_t g_kMaxBitStackSize = 64;

/*
  Let's say we want to push "True" to the bit stack at index 3.
  The mask is 1L << 3 = 0000000000000000000000000000000000000000000000000000000000001000
  bitStack | mask will ensure the 3rd bit is set to 1, while keeping the other bits unchanged.

  Let's say we want to push "False" to the bit stack at index 4.
  The mask is ~(1L << 4) = 1111111111111111111111111111111111111111111111111111111111101111
  bitStack & mask will ensure the 4th bit is set to 0, while keeping the other bits unchanged.
*/
__device__ void stackPush(uint64_t& bitStack, const uint8_t bitStackIdx, bool value)
{
    uint64_t mask = value? 1L << bitStackIdx : ~(1L << bitStackIdx);
    bitStack = value? bitStack | mask : bitStack & mask;
}

/*
  Let's say we want to get the top value of the bit stack at index 5.
  The mask is 1L << 5 = 0000000000000000000000000000000000000000000000000000000000100000
  bitStack & mask will ensure the 5th bit is kept, while the other bits are set to 0.
  This way, if the 5th bit is 1, then tmp > 0L, otherwise tmp is 0L.
*/
__device__ bool stackTop(const uint64_t bitStack, const uint8_t bitStackIdx)
{
    uint64_t mask = 1L << bitStackIdx;
    uint64_t tmp = bitStack & mask;
    return tmp > 0L;
}

__device__ bool matchOp(const AbmDataGpuOneField& reqAbmDataGpu,
                        const AbmDataGpuOneField& docAbmDataGpu,
                        const int reqIdx,
                        const int docIdx,
                        const CabmOp& op)
{
    // Get the offset iterators 
    int reqOffsetIter = 0;
    int docOffsetIter = 0;

    // Get the end offset iterators
    int reqOffsetEnd = reqAbmDataGpu.getNumVals(reqIdx);
    int docOffsetEnd = docAbmDataGpu.getNumVals(docIdx);

    bool rst = false;
    // We assume the req and doc data are sorted.
    while (reqOffsetIter < reqOffsetEnd && docOffsetIter < docOffsetEnd) // While the iterators are not at the end
    {
        // Get the values
        ABM_DATA_TYPE reqVal = reqAbmDataGpu.getVal(reqIdx, reqOffsetIter);
        ABM_DATA_TYPE docVal = docAbmDataGpu.getVal(docIdx, docOffsetIter);

        // If the values are equal, we have a match, so we can break.
        if (reqVal == docVal)
        {
            rst = true;
            break;
        }
        else if (reqVal < docVal) // If the req value is less than the doc value, we increment the req iterator.
        {
            reqOffsetIter++;
        }
        else // If the req value is greater than the doc value, we increment the doc iterator.
        {
            docOffsetIter++;
        }
    }

    if (op.isNegation()) // If the operand is negated, we negate the result.
    {
        rst = !rst;
    }

    return rst;
}

struct OperandKernelParam
{
    AbmDataGpuOneField reqAbmDataGpu;
    AbmDataGpuOneField docAbmDataGpu;
    CabmOp op;
    uint64_t reqIdx;
    uint64_t numDocs;
    uint64_t* d_bitStacks;
    uint8_t bitStackIdx;
};

__global__ void matchOpKernel(OperandKernelParam param)
{
    uint64_t docIdx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (docIdx < param.numDocs)
    {
        bool rst = matchOp(param.reqAbmDataGpu, param.docAbmDataGpu, param.reqIdx, docIdx, param.op);
        stackPush(param.d_bitStacks[docIdx], param.bitStackIdx, rst);
    }
}

struct OperatorKernelParam
{
    CabmOp op;
    uint64_t numPostfixOps;
    uint64_t* d_bitStacks;
    uint64_t numDocs;
    uint8_t bitStackIdx;
};

__global__ void operatorKernel(OperatorKernelParam param)
{
    uint64_t docIdx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (docIdx < param.numDocs)
    {
        uint64_t& bitStack = param.d_bitStacks[docIdx];
        bool rst1 = stackTop(bitStack, param.bitStackIdx - 1); // Get the first of first operand
        bool rst2 = stackTop(bitStack, param.bitStackIdx - 2); // Get the second of second operand
        bool rst = (param.op.getOpType() == CabmOpType::OPERATOR_AND) ? (rst1 & rst2) : (rst1 | rst2); // Apply the operator
        stackPush(bitStack, param.bitStackIdx - 2, rst); // Push the result to the bit stack
    }
}

__global__ void copyRstKernel(uint8_t* d_rst, uint64_t* d_bitStacks, uint64_t numDocs, int reqIdx)
{
    uint64_t docIdx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (docIdx < numDocs)
    {
        d_rst[reqIdx * numDocs + docIdx] = stackTop(d_bitStacks[docIdx], 0);
    }
}

void cabmGpu(CabmGpuParam& param)
{
    // -----------------
    // Reset time
    param.timeMsOperandKernel = 0;
    param.timeMsOperatorKernel = 0;
    param.timeMsCopyRstKernel = 0;
    param.timeMsTotal = 0;

    // -----------------
    // Start timer
    Timer timerTotal;
    timerTotal.tic();

    // -----------------
    // Set the block and grid size
    const int kBlockSize = 1024;
    const int kGridSize = (param.numDocs + kBlockSize - 1) / kBlockSize;

    for (uint32_t reqIdx = 0; reqIdx < param.numReqs; reqIdx++)
    {
        // -----------------
        // Initialize the bit stack index and set the bit stack to 0
        uint8_t currBitStackIdx = 0;
        CHECK_CUDA(cudaMemset(param.d_bitStacks, 0, param.numDocs * sizeof(uint64_t)));
        
        for (const auto& op : param.postfixOps)
        {
            if (op.isOperand())
            {
                // -----------------
                // Check if the bit stack index is greater than the maximum bit stack size
                if (currBitStackIdx >= g_kMaxBitStackSize)
                {
                    std::ostringstream oss;
                    oss << "currBitStackIdx is greater than g_kMaxBitStackCount: " << (int)currBitStackIdx
                        << " >= " << g_kMaxBitStackSize;
                    throw std::runtime_error(oss.str());
                }

                // -----------------
                // Create the operand kernel parameter
                OperandKernelParam operandKernelParam;
                operandKernelParam.reqAbmDataGpu = param.reqAbmDataGpuList.at(reqIdx);
                operandKernelParam.docAbmDataGpu = param.docAbmDataGpuList.at(reqIdx);
                operandKernelParam.op = op;
                operandKernelParam.reqIdx = reqIdx;
                operandKernelParam.numDocs = param.numDocs;
                operandKernelParam.d_bitStacks = param.d_bitStacks;
                operandKernelParam.bitStackIdx = currBitStackIdx;

                // -----------------
                // Launch the operand kernel
                Timer timerOperandKernel;
                timerOperandKernel.tic();
                if (op.getOpType() == CabmOpType::OPERAND_MATCH)
                {
                    matchOpKernel<<<kGridSize, kBlockSize>>>(operandKernelParam);
                }
                else
                {
                    std::ostringstream oss;
                    oss << "Invalid operator type: " << static_cast<int>(op.getOpType());
                    throw std::runtime_error(oss.str());
                }
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaGetLastError());
                param.timeMsOperandKernel += timerOperandKernel.tocMs();

                // -----------------
                // Increment the bit stack index
                // We push 1, so the net effect is +1
                currBitStackIdx++;
            }
            else if (op.isOperator())
            {
                // -----------------
                // Create the operator kernel parameter
                OperatorKernelParam operatorKernelParam;
                operatorKernelParam.op = op;
                operatorKernelParam.d_bitStacks = param.d_bitStacks;
                operatorKernelParam.bitStackIdx = currBitStackIdx;
                operatorKernelParam.numDocs = param.numDocs;

                // -----------------
                // Launch the operator kernel
                Timer timerOperatorKernel;
                timerOperatorKernel.tic();
                operatorKernel<<<kGridSize, kBlockSize>>>(operatorKernelParam);
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaGetLastError());
                param.timeMsOperatorKernel += timerOperatorKernel.tocMs();

                // -----------------
                // Decrement the bit stack index
                // We pop 2 and push 1, so the net effect is -1
                currBitStackIdx--;
            }
        }

        // -----------------
        // And the end of postfix evaluation, there should be exactly one value in the bit stack.
        if (currBitStackIdx != 1)
        {
            std::ostringstream oss;
            oss << "currBitStackIdx is not 1: " << (int)currBitStackIdx;
            throw std::runtime_error(oss.str());
        }

        // -----------------
        // Copy the result to the output
        Timer timerCopyRstKernel;
        timerCopyRstKernel.tic();
        copyRstKernel<<<kGridSize, kBlockSize>>>(param.d_rst, param.d_bitStacks, param.numDocs, reqIdx);
        CHECK_CUDA(cudaDeviceSynchronize())
        CHECK_CUDA(cudaGetLastError())
        param.timeMsCopyRstKernel += timerCopyRstKernel.tocMs();
    }

    // -----------------
    // Stop timer
    param.timeMsTotal = timerTotal.tocMs();
}

bool evaluatePostfixGpuWrapped(std::vector<CabmOp> postfix1D,
                               const std::vector<std::vector<ABM_DATA_TYPE>>& reqData2D,
                               const std::vector<std::vector<ABM_DATA_TYPE>>& docData2D)
{
    std::vector<AbmDataGpuOneField> reqAbmDataGpuList;
    std::vector<AbmDataGpuOneField> docAbmDataGpuList;
    for (int fieldIdx = 0; fieldIdx < reqData2D.size(); fieldIdx++)
    {
        reqAbmDataGpuList.push_back(AbmDataGpuOneField());
        docAbmDataGpuList.push_back(AbmDataGpuOneField());
        reqAbmDataGpuList.at(fieldIdx).init({reqData2D}, fieldIdx, true);
        docAbmDataGpuList.at(fieldIdx).init({docData2D}, fieldIdx, true);
    }

    uint8_t* d_rst;
    CHECK_CUDA(cudaMallocManaged(&d_rst, 1 * sizeof(uint8_t)));
    uint64_t* d_bitStacks;
    CHECK_CUDA(cudaMallocManaged(&d_bitStacks, 1 * sizeof(uint64_t)));

    CabmGpuParam param;
    param.d_rst = d_rst;
    param.d_bitStacks = d_bitStacks;
    param.numDocs = 1;
    param.numReqs = 1;
    param.postfixOps = postfix1D;
    param.reqAbmDataGpuList = reqAbmDataGpuList;
    param.docAbmDataGpuList = docAbmDataGpuList;

    cabmGpu(param);

    uint8_t rst = d_rst[0];

    CHECK_CUDA(cudaFree(d_rst));
    CHECK_CUDA(cudaFree(d_bitStacks));

    return rst;
}