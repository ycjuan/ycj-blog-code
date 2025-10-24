#include <cassert>
#include <cerrno>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <vector>

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

    if (op.isNegation())
    {
        rst = !rst;
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
        std::ostringstream oss;

        oss << "{";
        if (m_negation)
        {
            oss << "NOT ";
        }
        oss << "R" << m_reqFieldIdx << " ";

        switch (m_opType)
        {
        case CabmOpType::OPERAND_MATCH:
            oss << "MATCH";
            break;
        default:
            throw std::invalid_argument("Invalid operand type (1)");
        }
        oss << " D" << m_docFieldIdx << "}";
        return oss.str();
    }
    else
    {
        switch (m_opType)
        {
        case CabmOpType::OPERATOR_AND:
            return "AND";
        case CabmOpType::OPERATOR_OR:
            return "OR";
        case CabmOpType::OPERATOR_LEFT_PARENTHESIS:
            return "(";
        case CabmOpType::OPERATOR_RIGHT_PARENTHESIS:
            return ")";
        default:
            throw std::invalid_argument("Invalid operator type (2)");
        }
    }
}

std::vector<int> cabmCpu(const std::vector<CabmOp>& infixExpr,
                         const std::vector<std::vector<long>>& reqTbr2D,
                         const std::vector<std::vector<std::vector<long>>>& docTbr3D)
{
    std::vector<CabmOp> postfixExpr = infix2postfix(infixExpr);
    int numDocs = docTbr3D.size();

    std::vector<int> results(numDocs);
    for (int i = 0; i < numDocs; i++)
    {
        results.at(i) = evaluatePostfix(postfixExpr, reqTbr2D, docTbr3D.at(i));
    }

    return results;
}