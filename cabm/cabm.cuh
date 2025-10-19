#pragma once

#include <vector>

enum class CabmOpType
{
    CABM_OP_TYPE_ATTR_MATCH = 0,
    CABM_OP_TYPE_NOOP = 1000,
    CABM_OP_TYPE_AND = 1001,
    CABM_OP_TYPE_OR = 1002,
    CABM_OP_TYPE_NOT = 1003,
    CABM_OP_TYPE_LEFT_PARENTHESIS = 1004,
    CABM_OP_TYPE_RIGHT_PARENTHESIS = 1005
};

class CabmOp
{
public:
    CabmOp(int clause, long attr, CabmOpType type)
        : clause(clause)
        , type(type)
    {
    }
    CabmOp(CabmOpType type)
        : type(type)
    {
    }

    bool isOperand() const { return type < CabmOpType::CABM_OP_TYPE_NOOP; }
    int getPriority() const { return static_cast<int>(type); }
    int getClause() const { return clause; }
    bool isLeftParenthesis() const { return type == CabmOpType::CABM_OP_TYPE_LEFT_PARENTHESIS; }
    bool isRightParenthesis() const { return type == CabmOpType::CABM_OP_TYPE_RIGHT_PARENTHESIS; }
    CabmOpType getType() const { return type; }
private:
    const int clause = -1; // Only used when op type is CABM_OP_TYPE_ATTR
    const CabmOpType type;

};

std::vector<CabmOp> infix2postfix(std::vector<CabmOp> infix);
bool evaluatePostfix(std::vector<CabmOp> postfix1D, const std::vector<std::vector<long>>& docTbr2D);