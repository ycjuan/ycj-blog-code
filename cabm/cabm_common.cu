#include <cassert>
#include <cerrno>
#include <iostream>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <vector>

#include "cabm.cuh"

namespace // anonymous namespace
{

    constexpr bool g_kDebug = false;

}

// The code here is modified from:
// https://prepinsta.com/cpp-program/infix-to-postfix-conversion-using-stack/
std::vector<CabmOp> infix2postfix(std::vector<CabmOp> infix)
{
    std::vector<CabmOp> postfix;
    std::stack<CabmOp> s;

    for (auto op : infix)
    {
        if (g_kDebug)
        {
            std::cout << "Processing op: " << op.toString() << std::endl;
        }
        // if it is an operand, we can directly add it to the postfix expression
        if (op.isOperand())
        {
            if (g_kDebug)
            {
                std::cout << "[1] Adding operand to postfix: " << op.toString() << std::endl;
            }
            postfix.push_back(op);
        }
        // if it is a left parenthesis, we need to push it to the stack first
        else if (op.isLeftParenthesis())
        {
            if (g_kDebug)
            {
                std::cout << "[2] Adding left parenthesis to stack: " << op.toString() << std::endl;
            }
            s.push(op);
        }
        // when encountering a right parenthesis, pop the stack and push to postfix until we encounter a left
        // parenthesis
        else if (op.isRightParenthesis())
        {
            while (!s.top().isLeftParenthesis())
            {
                if (g_kDebug)
                {
                    std::cout << "[3] Adding operator to postfix: " << s.top().toString() << std::endl;
                }
                postfix.push_back(s.top());
                if (g_kDebug)
                {
                    std::cout << "[3] Popping from stack: " << s.top().toString() << std::endl;
                }
                s.pop();
            }
            if (g_kDebug)
            {
                std::cout << "[3] Popping left parenthesis from stack: " << s.top().toString() << std::endl;
            }
            s.pop();
        }
        // in this case, op is an operator. we keep popping the stack and push to postfix until we encounter a
        // non-operator
        else
        {
            while (!s.empty() && s.top().isOperator())
            {
                if (g_kDebug)
                {
                    std::cout << "[4] Adding operator to postfix: " << s.top().toString() << std::endl;
                }
                postfix.push_back(s.top());
                if (g_kDebug)
                {
                    std::cout << "[4] Popping from stack: " << s.top().toString() << std::endl;
                }
                s.pop();
            }
            if (g_kDebug)
            {
                std::cout << "[4] Adding operator to stack: " << op.toString() << std::endl;
            }
            s.push(op);
        }
    }
    // finally, pop all the remaining operators from the stack and push to postfix
    while (!s.empty())
    {
        if (g_kDebug)
        {
            std::cout << "[5] Adding operator to postfix: " << s.top().toString() << std::endl;
        }
        postfix.push_back(s.top());
        if (g_kDebug)
        {
            std::cout << "[5] Popping from stack: " << s.top().toString() << std::endl;
        }
        s.pop();
    }

    return postfix;
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
        case CabmOpType::LEFT_PARENTHESIS:
            return "(";
        case CabmOpType::RIGHT_PARENTHESIS:
            return ")";
        default:
            throw std::invalid_argument("Invalid operator type (2)");
        }
    }
}

bool canEarlyStop(bool stackTop, std::vector<CabmOp> postfix1D)
{
    // Create a stack of capacity equal to expression size
    std::stack<int> st;
    st.push(stackTop);

    // Scan all characters one by one
    for (auto op : postfix1D)
    {
        // If the scanned character is an operand
        // (number here), push it to the stack.
        if (op.isOperand())
        {
            st.push(!stackTop);
        }
        // If the scanned character is an operator,
        // pop two elements from stack apply the operator
        else
        {

            int rstA = st.empty() ? !stackTop : st.top();
            if (!st.empty()) 
            {
                st.pop();
            }
            int rstB = st.empty() ? !stackTop : st.top();
            if (!st.empty()) 
            {
                st.pop();
            }
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
    return stackTop == (bool)st.top();
}
