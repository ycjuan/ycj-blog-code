#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#include "data_struct.cuh"
#include "cabm.cuh"
#include "macro.cuh"
#include "util.cuh"

void test3a()
{
    const int kNumReqs = 1;
    const uint64_t kNumDocs = 1000;
    const int kNumFields = 6;
    const int kNumTrials = 100;
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

    {
        AbmDataGpu reqAbmDataGpu;
        reqAbmDataGpu.init(reqData3D, true);
        AbmDataGpu docAbmDataGpu;
        docAbmDataGpu.init(docData3D, true);

        uint8_t* d_rst;
        CHECK_CUDA(cudaMalloc(&d_rst, kNumReqs * kNumDocs * sizeof(uint8_t)));
        uint64_t* d_bitStacks;
        CHECK_CUDA(cudaMalloc(&d_bitStacks, kNumDocs * sizeof(uint64_t)));

        CabmGpuParam param;
        param.d_rst = d_rst;
        param.d_bitStacks = d_bitStacks;
        param.numDocs = kNumDocs;
        param.numReqs = kNumReqs;
        param.postfixOps = postfix;
        param.reqAbmDataGpu = reqAbmDataGpu;
        param.docAbmDataGpu = docAbmDataGpu;

        Timer timer;
        for (int trial = -3; trial < kNumTrials; trial++)
        {
            if (trial == 0)
            {
                timer.tic();
            }
            cabmGpu(param);
        }
        float timeMs = timer.tocMs() / kNumTrials;
        std::cout << "Time per trial: " << timeMs << " ms" << std::endl;

        std::vector<uint8_t> rstGpu(kNumReqs * kNumDocs);
        CHECK_CUDA(cudaMemcpy(rstGpu.data(), d_rst, kNumReqs * kNumDocs * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(d_rst));
        CHECK_CUDA(cudaFree(d_bitStacks));
    }
}

int main()
{
    test3a();
    return 0;
}