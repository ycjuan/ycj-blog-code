#include <cassert>
#include <cerrno>
#include <stack>
#include <vector>
#include <stdexcept>
#include <iostream>

#include "cabm.cuh"

// The code here is modified from: Code source:
// https://prepinsta.com/cpp-program/infix-to-postfix-conversion-using-stack/
std::vector<CabmOp> infix2postfix(std::vector<CabmOp> infix)
{
    std::vector<CabmOp> postfix;
    // using inbuilt stack< > from C++ stack library
    std::stack<CabmOp> s;

    for (auto op : infix)
    {

        // if operand add to the postfix expression
        if (op.isOperand())
        {
            postfix.push_back(op);
        }
        // if opening bracket then push the stack
        else if (op.isLeftParenthesis())
        {
            s.push(op);
        }
        // if closing bracket encounted then keep popping from stack until
        // closing a pair opening bracket is not encountered
        else if (op.isRightParenthesis())
        {
            while (!s.top().isLeftParenthesis())
            {
                postfix.push_back(s.top());
                s.pop();
            }
            s.pop();
        }
        else
        {
            while (!s.empty() && op.getPriority() <= s.top().getPriority())
            {
                postfix.push_back(s.top());
                s.pop();
            }
            s.push(op);
        }
    }
    while (!s.empty())
    {
        postfix.push_back(s.top());
        s.pop();
    }

    return postfix;
}

int evaluateOp(CabmOp& op,
               const std::vector<std::vector<long>>& reqTbr2D,
               const std::vector<std::vector<long>>& docTbr2D)
{
    const std::vector<long>& reqAttrs = reqTbr2D.at(op.getReqFieldIdx());
    const std::vector<long>& docAttrs = docTbr2D.at(op.getDocFieldIdx());

    // For CPU implementation, we will use this simple two-layer for loop.
    int rst = 0;
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

    return rst;
}

// the code is modified from https://www.geeksforgeeks.org/evaluation-of-postfix-expression/
bool evaluatePostfix(std::vector<CabmOp> postfix1D,
                     const std::vector<std::vector<long>>& reqTbr2D,
                     const std::vector<std::vector<long>>& docTbr2D)
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
            st.push(evaluateOp(op, reqTbr2D, docTbr2D));
        }
        // If the scanned character is an operator,
        // pop two elements from stack apply the operator
        else
        {
            int rstA = st.top();
            st.pop();
            int rstB = st.top();
            st.pop();
            switch (op.getType())
            {
            case CabmOpType::CABM_OP_TYPE_AND:
                st.push(int((bool)rstA & (bool)rstB));
                break;
            case CabmOpType::CABM_OP_TYPE_OR:
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

std::vector<int>
cabmCpu(const std::vector<CabmOp>& reqInfix1D, const std::vector<std::vector<long>>& reqTbr1D, const std::vector<std::vector<std::vector<long>>>& docTbr3D)
{

    std::vector<CabmOp> reqPostfix1D = infix2postfix(reqInfix1D);
    int numDocs = docTbr3D.size();
    int numClauses = docTbr3D[0].size();

    std::vector<int> rst1D(numDocs);
    for (int i = 0; i < numDocs; i++)
    {
        assert(docTbr3D[i].size() == numClauses);

        const std::vector<std::vector<long>>& docTbr2D = docTbr3D.at(i);
        bool finalRst = evaluatePostfix(reqPostfix1D, reqTbr1D, docTbr2D);
        rst1D.push_back(finalRst);
    }

    return rst1D;
}

std::string cabmExprToString(const std::vector<CabmOp>& expr)
{
    std::string rst;
    for (auto op : expr)
    {
        rst += op.toString() + " ";
    }
    rst.pop_back(); // remove the last space
    return rst;
}

std::string CabmOp::toString() const
{
    if (isOperand())
    {
        std::string opTypeStr;
        switch (m_opType)
        {
            case CabmOpType::CABM_OP_TYPE_ATTR_MATCH:
                opTypeStr = "MATCH";
                break;
            default:
                throw std::invalid_argument("Invalid operand type (1)");
        }
        return "[F" + std::to_string(m_reqFieldIdx) + "-" + opTypeStr + "]";
    }
    else
    {
        switch (m_opType)
        {
            case CabmOpType::CABM_OP_TYPE_AND:
                return "AND";
            case CabmOpType::CABM_OP_TYPE_OR:
                return "OR";
            case CabmOpType::CABM_OP_TYPE_LEFT_PARENTHESIS:
                return "(";
            case CabmOpType::CABM_OP_TYPE_RIGHT_PARENTHESIS:
                return ")";
            case CabmOpType::CABM_OP_TYPE_NOT:
                return "NOT";
            default:
                throw std::invalid_argument("Invalid operator type (2)");
        }
    }
}