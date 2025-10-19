#include <vector>

struct Msg {
    int i; // the index in the range of {0, ..., numDocs-1}
    int r; // the index in the range of {0, ..., numReqs-1}
    float bid;
    float score;
};

enum CabmOpType {
    CABM_OP_TYPE_ATTR_MATCH = 0,
    CABM_OP_TYPE_ATTR_NEGATION = 1,
    CABM_OP_TYPE_AND = 2,
    CABM_OP_TYPE_OR = 3,
    CABM_OP_TYPE_LEFT_PARENTHESIS = 4,
    CABM_OP_TYPE_RIGHT_PARENTHESIS = 5
};

struct CabmOp {
    CabmOp();
    CabmOp(int clause, long attr, CabmOpType type);
    CabmOp(CabmOpType type);
    int clause = -1; // Only used when op type is CABM_OP_TYPE_ATTR
    long attr = -1; // Only used when op type is CABM_OP_TYPE_ATTR
    bool result = false;
    bool isOperand = true; 
    CabmOpType type;
};

struct CabmGpuParam {
    long      *d_docTbrAttr;
    int       *d_docTbrOffsets;
    long       msgSize;
    int        numClauses; 
    int        docMaxNumTbrAttr;
    int        numDocs;
    int        numReqs = 1;
    Msg       *d_msgInit;
    Msg       *d_msg;
    Msg       *d_msgBuffer;
    float      timeMs = 0;
    CabmOp    *d_reqPostfixOp;
    int        reqPostfixExprLength;
    int        k = 0;
    long       offsetA;
    long       offsetB;
};

std::vector<CabmOp> infix2postfix(std::vector<CabmOp> infix);
bool evaluatePostfix(std::vector<CabmOp> postfix1D, const std::vector<std::vector<long>> &docTbr2D);
std::vector<Msg> cabmCpu(const std::vector<CabmOp> &reqInfix1D, const std::vector<std::vector<std::vector<long>>> &docTbr3D, const std::vector<Msg> &msg1D);
void cabmGpu(CabmGpuParam &param);
