#pragma once

#include <string>
#include <vector>

enum class CabmOpType
{
    OPERATOR_LEFT_PARENTHESIS = 0,
    OPERATOR_RIGHT_PARENTHESIS = 1,
    OPERATOR_AND = 2,
    OPERATOR_OR = 3,
    NOOP = 1000, // This is used to determine if an op is an operator or an operand by checking "isOperand == Op > NOOP?"
    OPERAND_MATCH = 1001,
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

    // Priority is used when converting infix to postfix. e.g., (a OR b AND c) -> (b c AND a OR)
    // Note if `priorityA < priorityB`, then `priorityA` has higher priority than `priorityB`
    int getPriority() const { return static_cast<int>(m_opType); }

    // Some convenient self-identifiers
    bool isOperand() const { return m_opType > CabmOpType::NOOP; }
    bool isOperator() const { return m_opType <= CabmOpType::NOOP; }
    bool isNegation() const { return m_negation; }
    bool isLeftParenthesis() const { return m_opType == CabmOpType::OPERATOR_LEFT_PARENTHESIS; }
    bool isRightParenthesis() const { return m_opType == CabmOpType::OPERATOR_RIGHT_PARENTHESIS; }

    // Convert to string
    std::string toString() const;

private:
    const int m_reqFieldIdx = -1; // Only used when op type is an operand
    const int m_docFieldIdx = -1; // Only used when op type is an operand
    const bool m_negation = false; // Only used when op type is an operand
    const CabmOpType m_opType = CabmOpType::NOOP;
};

std::string cabmExprToString(const std::vector<CabmOp>& expr);

std::vector<CabmOp> infix2postfix(std::vector<CabmOp> infix);

bool evaluatePostfix(std::vector<CabmOp> postfix1D,
                     const std::vector<std::vector<long>>& reqData2D,
                     const std::vector<std::vector<long>>& docData2D);

std::vector<int> cabmCpu(const std::vector<CabmOp>& infixExpr,
                         const std::vector<std::vector<long>>& reqData2D,
                         const std::vector<std::vector<std::vector<long>>>& docData3D);