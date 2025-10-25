#pragma once

#include <string>
#include <vector>

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
    int getReqFieldIdx() const { return m_reqFieldIdx; }
    int getDocFieldIdx() const { return m_docFieldIdx; }
    CabmOpType getOpType() const { return m_opType; }

    // Some convenient self-identifiers
    bool isOperand() const { return m_opType >= CabmOpType::OPERAND_MATCH; }
    bool isOperator() const { return m_opType < CabmOpType::OPERAND_MATCH && m_opType > CabmOpType::RIGHT_PARENTHESIS; }
    bool isNegation() const { return m_negation; }
    bool isLeftParenthesis() const { return m_opType == CabmOpType::LEFT_PARENTHESIS; }
    bool isRightParenthesis() const { return m_opType == CabmOpType::RIGHT_PARENTHESIS; }

    // Convert to string
    std::string toString() const;

private:
    const int m_reqFieldIdx = -1; // Only used when op type is an operand
    const int m_docFieldIdx = -1; // Only used when op type is an operand
    const bool m_negation = false; // Only used when op type is an operand
    const CabmOpType m_opType; // We do not assign a default value because this field is always assigned in the constructor.
};

std::string cabmExprToString(const std::vector<CabmOp>& expr);

std::vector<CabmOp> infix2postfix(std::vector<CabmOp> infix);

bool evaluatePostfix(std::vector<CabmOp> postfix1D,
                     const std::vector<std::vector<long>>& reqData2D,
                     const std::vector<std::vector<long>>& docData2D);

std::vector<int> cabmCpu(const std::vector<CabmOp>& infixExpr,
                         const std::vector<std::vector<long>>& reqData2D,
                         const std::vector<std::vector<std::vector<long>>>& docData3D);