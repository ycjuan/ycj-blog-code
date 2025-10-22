#include <cassert>
#include <iostream>
#include <vector>

#include "cabm.cuh"

void test1a()
{
    // infix = (F0 OR F1) AND ( (F3 OR F4) AND F5 )
    std::vector<CabmOp> infix = {
        CabmOp(CabmOpType::OPERATOR_LEFT_PARENTHESIS),
        CabmOp(0, 0, CabmOpType::OPERAND_MATCH),
        CabmOp(CabmOpType::OPERATOR_OR),
        CabmOp(1, 1, CabmOpType::OPERAND_MATCH),
        CabmOp(CabmOpType::OPERATOR_RIGHT_PARENTHESIS),
        CabmOp(CabmOpType::OPERATOR_AND),
        CabmOp(CabmOpType::OPERATOR_LEFT_PARENTHESIS),
        CabmOp(CabmOpType::OPERATOR_LEFT_PARENTHESIS),
        CabmOp(3, 3, CabmOpType::OPERAND_MATCH),
        CabmOp(CabmOpType::OPERATOR_OR),
        CabmOp(4, 4, CabmOpType::OPERAND_MATCH),
        CabmOp(CabmOpType::OPERATOR_RIGHT_PARENTHESIS),
        CabmOp(CabmOpType::OPERATOR_AND),
        CabmOp(5, 5, CabmOpType::OPERAND_MATCH),
        CabmOp(CabmOpType::OPERATOR_RIGHT_PARENTHESIS),
    };

    std::cout << "Infix: " << cabmExprToString(infix) << std::endl;

    assert(cabmExprToString(infix)
           == "( [F0-MATCH] OR [F1-MATCH] ) AND ( ( [F3-MATCH] OR [F4-MATCH] ) AND [F5-MATCH] )");

    std::vector<CabmOp> postfix = infix2postfix(infix);

    std::cout << "Postfix: " << cabmExprToString(postfix) << std::endl;

    assert(cabmExprToString(postfix) == "[F0-MATCH] [F1-MATCH] OR [F3-MATCH] [F4-MATCH] OR [F5-MATCH] AND AND");

    {
        std::vector<std::vector<long>> reqTbr2D = { { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 5 } };
        std::vector<std::vector<long>> docTbr2D = { { 0 }, { -1 }, { -2 }, { 3 }, { -4 }, { 5 } };
        bool rst = evaluatePostfix(postfix, reqTbr2D, docTbr2D);
        assert(rst == true);
    }
}

int main()
{
    test1a();
    return 0;
}