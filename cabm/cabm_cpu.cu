#include <cassert>
#include <cerrno>
#include <stack>
#include <stdexcept>
#include <vector>

#include "cabm.cuh"
#include "data_struct.cuh"

int evaluateOp(CabmOp& op,
               const std::vector<std::vector<ABM_DATA_TYPE>>& reqData2D,
               const std::vector<std::vector<ABM_DATA_TYPE>>& docData2D)
{
    const std::vector<ABM_DATA_TYPE>& reqAttrs = reqData2D.at(op.getReqFieldIdx());
    const std::vector<ABM_DATA_TYPE>& docAttrs = docData2D.at(op.getDocFieldIdx());

    int rst = 0;

    // For CPU implementation, we will use this simple two-layer for loop.
    if (op.getOpType() == CabmOpType::OPERAND_MATCH)
    {
        for (auto reqAttr : reqAttrs)
        {
            for (auto docAttr : docAttrs)
            {
                if (reqAttr == docAttr)
                {
                    rst = 1;
                    break;
                }
            }
        }
    }
    else
    {
        throw std::invalid_argument("Invalid operator type: " + std::to_string(static_cast<int>(op.getOpType())));
    }

    if (op.isNegation())
    {
        rst = !rst;
    }

    return rst;
}

// the code is modified from https://www.geeksforgeeks.org/evaluation-of-postfix-expression/
bool evaluatePostfix(std::vector<CabmOp> postfix1D,
                     const std::vector<std::vector<ABM_DATA_TYPE>>& reqData2D,
                     const std::vector<std::vector<ABM_DATA_TYPE>>& docData2D)
{

    // Create a stack of capacity equal to expression size
    std::stack<int> st;

    // Scan all characters one by one
    for (auto op : postfix1D)
    {
        // If the scanned character is an operand
        // (number here), push it to the stack.
        if (op.isOperand())
        {
            st.push(evaluateOp(op, reqData2D, docData2D));
        }
        // If the scanned character is an operator,
        // pop two elements from stack apply the operator
        else
        {
            int rstA = st.top();
            st.pop();
            int rstB = st.top();
            st.pop();
            switch (op.getOpType())
            {
            case CabmOpType::OPERATOR_AND:
                st.push(int((bool)rstA & (bool)rstB));
                break;
            case CabmOpType::OPERATOR_OR:
                st.push(int((bool)rstA | (bool)rstB));
                break;
            default:
                assert(false); // This should not happen
                break;
            }
        }
    }
    return (bool)st.top();
}

std::vector<std::vector<uint8_t>> cabmCpu(const std::vector<CabmOp>& infixExpr,
                                          const std::vector<std::vector<std::vector<ABM_DATA_TYPE>>>& reqData3D,
                                          const std::vector<std::vector<std::vector<ABM_DATA_TYPE>>>& docData3D)
{
    std::vector<CabmOp> postfixExpr = infix2postfix(infixExpr);

    std::vector<std::vector<uint8_t>> rst2D;
    rst2D.resize(reqData3D.size());
    for (uint32_t reqIdx = 0; reqIdx < reqData3D.size(); reqIdx++)
    {
        rst2D.at(reqIdx).resize(docData3D.size());
        for (uint32_t docIdx = 0; docIdx < docData3D.size(); docIdx++)
        {
            rst2D.at(reqIdx).at(docIdx) = evaluatePostfix(postfixExpr, reqData3D.at(reqIdx), docData3D.at(docIdx));
        }
    }
    return rst2D;
}