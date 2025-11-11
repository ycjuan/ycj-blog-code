#pragma once

#include <string>
#include <vector>

#include "data_struct.cuh"

enum class CabmOpType
{
    LEFT_PARENTHESIS = 0,
    RIGHT_PARENTHESIS = 1,
    OPERATOR_AND = 2,
    OPERATOR_OR = 3,
    OPERAND_MATCH = 100,
};

class CabmOp
{
public:
    CabmOp() = default; // Default constructor
    CabmOp(int reqFieldIdx, int docFieldIdx, CabmOpType type, bool negation = false) // Constructor for operands
        : m_reqFieldIdx(reqFieldIdx)
        , m_docFieldIdx(docFieldIdx)
        , m_opType(type)
        , m_negation(negation)
    {
    }
    CabmOp(CabmOpType type) // Constructor for operators
        : m_opType(type)
    {
    }

    // Getters
    __device__ __host__ int getReqFieldIdx() const { return m_reqFieldIdx; }
    __device__ __host__ int getDocFieldIdx() const { return m_docFieldIdx; }
    __device__ __host__ CabmOpType getOpType() const { return m_opType; }

    // Some convenient self-identifiers
    bool isOperand() const { return m_opType >= CabmOpType::OPERAND_MATCH; }
    bool isOperator() const { return m_opType < CabmOpType::OPERAND_MATCH && m_opType > CabmOpType::RIGHT_PARENTHESIS; }
    __device__ __host__ bool isNegation() const { return m_negation; }
    bool isLeftParenthesis() const { return m_opType == CabmOpType::LEFT_PARENTHESIS; }
    bool isRightParenthesis() const { return m_opType == CabmOpType::RIGHT_PARENTHESIS; }

    // Convert to string
    std::string toString() const;

private:
    int m_reqFieldIdx = -1; // Only used when op type is an operand
    int m_docFieldIdx = -1; // Only used when op type is an operand
    bool m_negation = false; // Only used when op type is an operand
    CabmOpType m_opType; 
};

std::string cabmExprToString(const std::vector<CabmOp>& expr);

std::vector<CabmOp> infix2postfix(std::vector<CabmOp> infix);

bool evaluatePostfix(std::vector<CabmOp> postfix1D,
                     const std::vector<std::vector<ABM_DATA_TYPE>>& reqData2D,
                     const std::vector<std::vector<ABM_DATA_TYPE>>& docData2D);

bool evaluatePostfixGpuWrapped(std::vector<CabmOp> postfix1D,
                               const std::vector<std::vector<ABM_DATA_TYPE>>& reqData2D,
                               const std::vector<std::vector<ABM_DATA_TYPE>>& docData2D);

std::vector<std::vector<uint8_t>> cabmCpu(const std::vector<CabmOp>& infixExpr,
                                          const std::vector<std::vector<std::vector<ABM_DATA_TYPE>>>& reqData3D,
                                          const std::vector<std::vector<std::vector<ABM_DATA_TYPE>>>& docData3D);

struct CabmGpuParam
{
    std::vector<AbmDataGpu> reqAbmDataGpuList;
    std::vector<AbmDataGpu> docAbmDataGpuList;
    std::vector<CabmOp> postfixOps;
    uint64_t* d_bitStacks;
    uint64_t numDocs;
    uint64_t numReqs;
    uint8_t* d_rst;
    uint8_t* d_canEarlyStop;
    float timeMsOperandKernel = 0;
    float timeMsOperatorKernel = 0;
    float timeMsCopyRstKernel = 0;
    float timeMsTotal = 0;
    bool enableEarlyStop = false;
};

void cabmGpu(CabmGpuParam& param);

bool canEarlyStop(bool stackTop, std::vector<CabmOp> postfix1D);