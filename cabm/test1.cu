#include <iostream>
#include <vector>

#include "cabm.cuh"

int main()
{
    std::vector<CabmOp> infix = {
        CabmOp(0, 1, CABM_OP_TYPE_ATTR_MATCH),
        CabmOp(0, 2, CABM_OP_TYPE_ATTR_MATCH),
        CabmOp(0, 3, CABM_OP_TYPE_ATTR_MATCH),
        CabmOp(0, 4, CABM_OP_TYPE_ATTR_MATCH),
        CabmOp(0, 5, CABM_OP_TYPE_ATTR_MATCH),
        CabmOp(0, 6, CABM_OP_TYPE_ATTR_MATCH),
    };
    std::cout << "Hello, World!" << std::endl;
    return 0;
}