#include <cassert>
#include <iostream>
#include <vector>

#include "cabm.cuh"

void test1a()
{
    // infix = (F0 OR F1) AND ( (F3 OR F4) AND F5 )
    std::vector<CabmOp> infix = {
        CabmOp(CabmOpType::LEFT_PARENTHESIS),
        CabmOp(0, 0, CabmOpType::OPERAND_MATCH),
        CabmOp(CabmOpType::OPERATOR_OR),
        CabmOp(1, 1, CabmOpType::OPERAND_MATCH),
        CabmOp(CabmOpType::RIGHT_PARENTHESIS),
        CabmOp(CabmOpType::OPERATOR_AND),
        CabmOp(CabmOpType::LEFT_PARENTHESIS),
        CabmOp(CabmOpType::LEFT_PARENTHESIS),
        CabmOp(3, 3, CabmOpType::OPERAND_MATCH, true),
        CabmOp(CabmOpType::OPERATOR_OR),
        CabmOp(4, 4, CabmOpType::OPERAND_MATCH),
        CabmOp(CabmOpType::RIGHT_PARENTHESIS),
        CabmOp(CabmOpType::OPERATOR_AND),
        CabmOp(5, 5, CabmOpType::OPERAND_MATCH),
        CabmOp(CabmOpType::RIGHT_PARENTHESIS),
    };

    std::cout << "Infix: " << cabmExprToString(infix) << std::endl;

    assert(cabmExprToString(infix)
           == "( {R0 MATCH D0} OR {R1 MATCH D1} ) AND ( ( {NOT R3 MATCH D3} OR {R4 MATCH D4} ) AND {R5 MATCH D5} )");

    std::vector<CabmOp> postfix = infix2postfix(infix);

    std::cout << "Postfix: " << cabmExprToString(postfix) << std::endl;

    assert(cabmExprToString(postfix) == "{R0 MATCH D0} {R1 MATCH D1} OR {NOT R3 MATCH D3} {R4 MATCH D4} OR {R5 MATCH D5} AND AND");

    {
        std::vector<std::vector<ABM_DATA_TYPE>> reqTbr2D = { { 0 }, { 1 }, { 2 }, { -3 }, { 4 }, { 5 } };
        std::vector<std::vector<ABM_DATA_TYPE>> docTbr2D = { { 0 }, { -1 }, { -2 }, { 3 }, { -4 }, { 5 } };
        bool rstCpu = evaluatePostfix(postfix, reqTbr2D, docTbr2D);
        std::cout << "test1a_1, cpu rst: " << rstCpu << std::endl;
        assert(rstCpu == true);

        bool rstGpu = evaluatePostfixGpuWrapped(postfix, reqTbr2D, docTbr2D);
        std::cout << "test1a_1, gpu rst: " << rstGpu << std::endl;
        assert(rstGpu == rstCpu);
    }

    {
        std::vector<std::vector<ABM_DATA_TYPE>> reqTbr2D = { { 0 }, { 1 }, { 2 }, { -3 }, { 4 }, { -5 } };
        std::vector<std::vector<ABM_DATA_TYPE>> docTbr2D = { { 0 }, { -1 }, { -2 }, { 3 }, { -4 }, { 5 } };
        bool rstCpu = evaluatePostfix(postfix, reqTbr2D, docTbr2D);
        std::cout << "test1a_2, cpu rst: " << rstCpu << std::endl;
        assert(rstCpu == false);

        bool rstGpu = evaluatePostfixGpuWrapped(postfix, reqTbr2D, docTbr2D);
        std::cout << "test1a_2, gpu rst: " << rstGpu << std::endl;
        assert(rstGpu == rstCpu);
    }
}

void test1b()
{
    // infix = (F0 OR F1) AND ( (F3 OR F4) AND F5 )
    std::vector<CabmOp> infix = {
        CabmOp(CabmOpType::LEFT_PARENTHESIS),
        CabmOp(0, 0, CabmOpType::OPERAND_MATCH),
        CabmOp(CabmOpType::OPERATOR_OR),
        CabmOp(1, 1, CabmOpType::OPERAND_MATCH),
        CabmOp(CabmOpType::RIGHT_PARENTHESIS)
    };

    std::cout << "Infix: " << cabmExprToString(infix) << std::endl;

    assert(cabmExprToString(infix)
           == "( {R0 MATCH D0} OR {R1 MATCH D1} )");

    std::vector<CabmOp> postfix = infix2postfix(infix);

    std::cout << "Postfix: " << cabmExprToString(postfix) << std::endl;

    assert(cabmExprToString(postfix) == "{R0 MATCH D0} {R1 MATCH D1} OR");

    {
        std::vector<std::vector<ABM_DATA_TYPE>> reqTbr2D = { { 0 }, { 1 } };
        std::vector<std::vector<ABM_DATA_TYPE>> docTbr2D = { { 0 }, { 1 } };
        bool rst = evaluatePostfix(postfix, reqTbr2D, docTbr2D);
        assert(rst == true);

        bool rstGpu = evaluatePostfixGpuWrapped(postfix, reqTbr2D, docTbr2D);
        std::cout << "rstGpu: " << rstGpu << std::endl;
        assert(rstGpu == rst);
    }
}

void test1c()
{
    // infix = (F0 OR F1) AND ( (F3 OR F4) AND F5 )
    std::vector<CabmOp> infix = {
        CabmOp(CabmOpType::LEFT_PARENTHESIS),
        CabmOp(0, 0, CabmOpType::OPERAND_MATCH),
        CabmOp(CabmOpType::OPERATOR_OR),
        CabmOp(1, 1, CabmOpType::OPERAND_MATCH),
        CabmOp(CabmOpType::RIGHT_PARENTHESIS),
        CabmOp(CabmOpType::OPERATOR_AND),
        CabmOp(CabmOpType::LEFT_PARENTHESIS),
        CabmOp(3, 3, CabmOpType::OPERAND_MATCH, true),
        CabmOp(CabmOpType::OPERATOR_OR),
        CabmOp(4, 4, CabmOpType::OPERAND_MATCH),
        CabmOp(CabmOpType::RIGHT_PARENTHESIS),
    };

    std::cout << "Infix: " << cabmExprToString(infix) << std::endl;

    assert(cabmExprToString(infix)
           == "( {R0 MATCH D0} OR {R1 MATCH D1} ) AND ( {NOT R3 MATCH D3} OR {R4 MATCH D4} )");

    std::vector<CabmOp> postfix = infix2postfix(infix);

    std::cout << "Postfix: " << cabmExprToString(postfix) << std::endl;

    assert(cabmExprToString(postfix) == "{R0 MATCH D0} {R1 MATCH D1} OR {NOT R3 MATCH D3} {R4 MATCH D4} OR AND");

    {
        std::vector<std::vector<ABM_DATA_TYPE>> reqTbr2D = { { 0 }, { 1 }, {2}, { -3 }, { 4 } };
        std::vector<std::vector<ABM_DATA_TYPE>> docTbr2D = { { 0 }, { -1 }, {2}, { 3 }, { -4 } };
        bool rstCpu = evaluatePostfix(postfix, reqTbr2D, docTbr2D);
        std::cout << "test1c, cpu rst: " << rstCpu << std::endl;
        assert(rstCpu == true);

        bool rstGpu = evaluatePostfixGpuWrapped(postfix, reqTbr2D, docTbr2D);
        std::cout << "test1c, gpu rst: " << rstGpu << std::endl;
        assert(rstGpu == rstCpu);
    }
}

int main()
{
    test1a();
    test1b();
    test1c();
    return 0;
}