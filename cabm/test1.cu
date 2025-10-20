#include <iostream>
#include <vector>
#include <cassert>

#include "cabm.cuh"

void test1a()
{
    // infix = (F0 OR F1) AND ( (F3 OR F4) AND F5 )
    std::vector<CabmOp> infix = {
        CabmOp(CabmOpType::CABM_OP_TYPE_LEFT_PARENTHESIS),
        CabmOp(0, CabmOpType::CABM_OP_TYPE_ATTR_MATCH),
        CabmOp(CabmOpType::CABM_OP_TYPE_OR),
        CabmOp(1, CabmOpType::CABM_OP_TYPE_ATTR_MATCH),
        CabmOp(CabmOpType::CABM_OP_TYPE_RIGHT_PARENTHESIS),
        CabmOp(CabmOpType::CABM_OP_TYPE_AND),
        CabmOp(CabmOpType::CABM_OP_TYPE_LEFT_PARENTHESIS),
        CabmOp(CabmOpType::CABM_OP_TYPE_LEFT_PARENTHESIS),
        CabmOp(3, CabmOpType::CABM_OP_TYPE_ATTR_MATCH),
        CabmOp(CabmOpType::CABM_OP_TYPE_OR),
        CabmOp(4, CabmOpType::CABM_OP_TYPE_ATTR_MATCH),
        CabmOp(CabmOpType::CABM_OP_TYPE_RIGHT_PARENTHESIS),
        CabmOp(CabmOpType::CABM_OP_TYPE_AND),
        CabmOp(5, CabmOpType::CABM_OP_TYPE_ATTR_MATCH),
        CabmOp(CabmOpType::CABM_OP_TYPE_RIGHT_PARENTHESIS),
    };

    std::cout << "Infix: " << cabmExprToString(infix) << std::endl;

    assert(cabmExprToString(infix) == "( [F0-MATCH] OR [F1-MATCH] ) AND ( ( [F3-MATCH] OR [F4-MATCH] ) AND [F5-MATCH] )");
    
    std::vector<CabmOp> postfix = infix2postfix(infix);
    
    std::cout << "Postfix: " << cabmExprToString(postfix) << std::endl;

    assert(cabmExprToString(postfix) == "[F0-MATCH] [F1-MATCH] OR [F3-MATCH] [F4-MATCH] OR [F5-MATCH] AND AND");
}

int main()
{
    test1a();
    return 0;
}