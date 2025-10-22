#pragma once

#include <string>
#include <vector>

enum class CabmOpType
{
    CABM_OP_TYPE_LEFT_PARENTHESIS = 0,
    CABM_OP_TYPE_RIGHT_PARENTHESIS = 1,
    CABM_OP_TYPE_NOT = 2,
    CABM_OP_TYPE_AND = 3,
    CABM_OP_TYPE_OR = 4,
    CABM_OP_TYPE_NOOP = 1000,
    CABM_OP_TYPE_ATTR_MATCH = 1001,
};

class CabmOp
{
public:
    CabmOp(int reqFieldIdx, int docFieldIdx, CabmOpType type)
        : m_reqFieldIdx(reqFieldIdx)
        , m_docFieldIdx(docFieldIdx)
        , m_opType(type)
    {
    }
    CabmOp(CabmOpType type)
        : m_opType(type)
    {
    }

    bool isOperand() const { return m_opType > CabmOpType::CABM_OP_TYPE_NOOP; }
    int getPriority() const { return static_cast<int>(m_opType); }
    int getReqFieldIdx() const { return m_reqFieldIdx; }
    int getDocFieldIdx() const { return m_docFieldIdx; }
    bool isLeftParenthesis() const { return m_opType == CabmOpType::CABM_OP_TYPE_LEFT_PARENTHESIS; }
    bool isRightParenthesis() const { return m_opType == CabmOpType::CABM_OP_TYPE_RIGHT_PARENTHESIS; }
    CabmOpType getType() const { return m_opType; }
    std::string toString() const;

private:
    const int m_reqFieldIdx = -1; // Only used when op type is an operand
    const int m_docFieldIdx = -1; // Only used when op type is an operand
    const CabmOpType m_opType;
};

std::string cabmExprToString(const std::vector<CabmOp>& expr);
std::vector<CabmOp> infix2postfix(std::vector<CabmOp> infix);
bool evaluatePostfix(std::vector<CabmOp> postfix1D,
                     const std::vector<std::vector<long>>& reqTbr2D,
                     const std::vector<std::vector<long>>& docTbr2D);