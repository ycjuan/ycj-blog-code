#include <cassert>
#include <iostream>
#include <vector>

#include "data_struct.cuh"
#include "cabm.cuh"

void test3a()
{
    const int kNumReqs = 1;
    const int kNumDocs = 10;
    const int kNumFields = 6;
    const std::vector<int> kNumValsPerFieldMin = { 2, 2, 2, 2, 2, 2 };
    const std::vector<int> kNumValsPerFieldMax = { 10, 10, 10, 10, 10, 10 };

    const auto reqData3D = genRandData3D(kNumReqs, kNumFields, kNumValsPerFieldMin, kNumValsPerFieldMax);
    const auto docData3D = genRandData3D(kNumDocs, kNumFields, kNumValsPerFieldMin, kNumValsPerFieldMax);

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

    std::vector<CabmOp> postfix = infix2postfix(infix);

    const auto rst2D = cabmCpu(infix, reqData3D, docData3D);
}

int main()
{
    test3a();
    return 0;
}