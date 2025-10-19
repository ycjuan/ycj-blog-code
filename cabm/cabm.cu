#include <iostream>
#include <stack>
#include <string>
#include <vector>
#include <cassert>

#ifndef CABM_H
#define CABM_H
#include "cabm.cuh"
#endif

using namespace std;

CabmOp::CabmOp() {}

CabmOp::CabmOp(int clause, long attr, CabmOpType type) : clause(clause), attr(attr), type(type) {
    if (type == CABM_OP_TYPE_ATTR_MATCH || type == CABM_OP_TYPE_ATTR_NEGATION)
        isOperand = true;
    else
        isOperand = false;
}

CabmOp::CabmOp(CabmOpType type) : type(type) {
    if (type == CABM_OP_TYPE_ATTR_MATCH || type == CABM_OP_TYPE_ATTR_NEGATION)
        isOperand = true;
    else
        isOperand = false;
}

int priority(CabmOp op) 
{
    if (op.type == CABM_OP_TYPE_AND || op.type == CABM_OP_TYPE_OR)
        return 1;

    return 0;
}

// The code here is modified from: Code source: https://prepinsta.com/cpp-program/infix-to-postfix-conversion-using-stack/
vector<CabmOp> infix2postfix(vector<CabmOp> infix)
{
    vector<CabmOp> postfix;
    // using inbuilt stack< > from C++ stack library
    stack<CabmOp> s;

    for (auto op : infix) 
    {

        // if operand add to the postfix expression
        if (op.type == CABM_OP_TYPE_ATTR_MATCH || op.type == CABM_OP_TYPE_ATTR_NEGATION) 
        {
            postfix.push_back(op);
        }
        // if opening bracket then push the stack
        else if (op.type == CABM_OP_TYPE_LEFT_PARENTHESIS) 
        {
            s.push(op);
        }
        // if closing bracket encounted then keep popping from stack until
        // closing a pair opening bracket is not encountered
        else if (op.type == CABM_OP_TYPE_RIGHT_PARENTHESIS) 
        {
            while (s.top().type != CABM_OP_TYPE_LEFT_PARENTHESIS) 
            {
                postfix.push_back(s.top());
                s.pop();
            }
            s.pop();
        }
        else
        {
            while (!s.empty() && priority(op) <= priority(s.top())){
                postfix.push_back(s.top());
                s.pop();
            }
            s.push(op);
        }
    }
    while (!s.empty()) {
        postfix.push_back(s.top());
        s.pop();
    }

    return postfix;
}

int evaluateOp(CabmOp &op, const vector<vector<long>> &docTbr2D) {
    int rst = 0;
    for (auto docAttr : docTbr2D[op.clause]) {
        if (op.attr == docAttr) {
            rst = 1;
            break;
        }
    }
    if (op.type == CABM_OP_TYPE_ATTR_NEGATION)
        rst = !rst;
    return rst;
}

// the code is modified from https://www.geeksforgeeks.org/evaluation-of-postfix-expression/
bool evaluatePostfix(vector<CabmOp> postfix1D, const vector<vector<long>> &docTbr2D) {

    // Create a stack of capacity equal to expression size
    stack<int> st;

    // Scan all characters one by one
    for (auto op : postfix1D) {

        // If the scanned character is an operand
        // (number here), push it to the stack.
        if ( op.isOperand ) 
            st.push( evaluateOp(op, docTbr2D) );

        // If the scanned character is an operator,
        // pop two elements from stack apply the operator
        else {
            int rstA = st.top();
            st.pop();
            int rstB = st.top();
            st.pop();
            switch (op.type) {
            case CABM_OP_TYPE_AND:
                st.push( int( (bool)rstA & (bool)rstB ) );
                break;
            case CABM_OP_TYPE_OR:
                st.push( int( (bool)rstA & (bool)rstB ) );
                break;
            default:
                assert(false);
                break;
            }
        }
    }
    return (bool)st.top();
}

vector<Msg> cabmCpu(const vector<CabmOp> &reqInfix1D, const vector<vector<vector<long>>> &docTbr3D, const vector<Msg> &msg1D) {

    vector<CabmOp> reqPostfix1D = infix2postfix(reqInfix1D);
    int numDocs = docTbr3D.size();
    int numClauses = docTbr3D[0].size();

    vector<Msg> rst1D;
    for (auto msg : msg1D) {
        int i = msg.i;
        assert(docTbr3D[i].size() == numClauses);

        const vector<vector<long>> &docTbr2D = docTbr3D[i];
        bool finalRst = evaluatePostfix(reqPostfix1D, docTbr2D);
        if (finalRst) {
            msg.score = 1.0;
            rst1D.push_back(msg);
        }
    }
    
    return rst1D;
}

