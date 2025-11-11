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
    const uint64_t kNumDocs = 1000000;
    const int kNumFields = 6;
    const int kNumTrials = 100;
    const std::vector<int> kNumValsPerFieldMin = { 2, 2, 2, 2, 2, 2 };
    const std::vector<int> kNumValsPerFieldMax = { 10, 10, 10, 10, 10, 10 };
    const std::vector<int> kCardinalities = { 100, 100, 100, 100, 100, 100 };

    const auto reqData3D
        = genRandData3D(kNumReqs, kNumFields, kNumValsPerFieldMin, kNumValsPerFieldMax, kCardinalities);
    const auto docData3D
        = genRandData3D(kNumDocs, kNumFields, kNumValsPerFieldMin, kNumValsPerFieldMax, kCardinalities);

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

    std::vector<std::vector<uint8_t>> rstGpu2D(kNumReqs, std::vector<uint8_t>(kNumDocs));
    {
        std::vector<AbmDataGpu> reqAbmDataGpuList;
        std::vector<AbmDataGpu> docAbmDataGpuList;
        for (int fieldIdx = 0; fieldIdx < kNumFields; fieldIdx++)
        {
            reqAbmDataGpuList.push_back(AbmDataGpu());
            docAbmDataGpuList.push_back(AbmDataGpu());
            reqAbmDataGpuList.at(fieldIdx).init({reqData3D}, fieldIdx, true);
            docAbmDataGpuList.at(fieldIdx).init({docData3D}, fieldIdx, true);
        }

        uint8_t* d_rst;
        CHECK_CUDA(cudaMalloc(&d_rst, kNumReqs * kNumDocs * sizeof(uint8_t)));
        uint64_t* d_bitStacks;
        CHECK_CUDA(cudaMalloc(&d_bitStacks, kNumDocs * sizeof(uint64_t)));
        uint8_t* d_canEarlyStop;
        CHECK_CUDA(cudaMalloc(&d_canEarlyStop, kNumDocs * sizeof(uint8_t)));

        CabmGpuParam param;
        param.d_rst = d_rst;
        param.d_bitStacks = d_bitStacks;
        param.numDocs = kNumDocs;
        param.numReqs = kNumReqs;
        param.postfixOps = postfix;
        param.reqAbmDataGpuList = reqAbmDataGpuList;
        param.docAbmDataGpuList = docAbmDataGpuList;
        param.d_canEarlyStop = d_canEarlyStop;
        param.enableEarlyStop = true;

        Timer timer;
        float timeMsOperandKernel = 0;
        float timeMsOperatorKernel = 0;
        float timeMsCopyRstKernel = 0;
        float timeMsTotal = 0;
        float timeMsTotalOuter = 0;
        for (int trial = -3; trial < kNumTrials; trial++)
        {
            if (trial == 0)
            {
                timer.tic();
            }
            cabmGpu(param);
            if (trial >= 0)
            {
                timeMsOperandKernel += param.timeMsOperandKernel;
                timeMsOperatorKernel += param.timeMsOperatorKernel;
                timeMsCopyRstKernel += param.timeMsCopyRstKernel;
                timeMsTotal += param.timeMsTotal;
            }
        }
        timeMsTotalOuter = timer.tocMs() / kNumTrials;
        timeMsOperandKernel /= kNumTrials;
        timeMsOperatorKernel /= kNumTrials;
        timeMsCopyRstKernel /= kNumTrials;
        timeMsTotal /= kNumTrials;
        std::cout << "Time total outer: " << timeMsTotalOuter << " ms" << std::endl;
        std::cout << "Time total inner: " << timeMsTotal << " ms" << std::endl;
        std::cout << "Time operand kernel: " << timeMsOperandKernel << " ms" << std::endl;
        std::cout << "Time operator kernel: " << timeMsOperatorKernel << " ms" << std::endl;
        std::cout << "Time copy rst kernel: " << timeMsCopyRstKernel << " ms" << std::endl;

        std::vector<uint8_t> rstGpu(kNumReqs * kNumDocs);
        CHECK_CUDA(cudaMemcpy(rstGpu.data(), d_rst, kNumReqs * kNumDocs * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(d_rst));
        CHECK_CUDA(cudaFree(d_bitStacks));
        CHECK_CUDA(cudaFree(d_canEarlyStop));
        for (int reqIdx = 0; reqIdx < kNumReqs; reqIdx++)
        {
            for (int docIdx = 0; docIdx < kNumDocs; docIdx++)
            {
                rstGpu2D.at(reqIdx).at(docIdx) = rstGpu.at(reqIdx * kNumDocs + docIdx);
            }
        }
    }

    // ----------------
    // Compare rst2D and rstGpu2D
    {
        int numCTGF = 0;
        int numCFGT = 0;
        for (int reqIdx = 0; reqIdx < kNumReqs; reqIdx++)
        {
            for (int docIdx = 0; docIdx < kNumDocs; docIdx++)
            {
                int rstCpu = rst2D.at(reqIdx).at(docIdx);
                int rstGpu = rstGpu2D.at(reqIdx).at(docIdx);
                if (rstCpu != rstGpu)
                {
                    if (rstCpu == 1 && rstGpu == 0)
                    {
                        numCTGF++;
                    }
                    else
                    {
                        numCFGT++;
                    }
                    //std::cout << "Error at (" << reqIdx << ", " << docIdx << "): " << rstCpu << " != " << rstGpu << std::endl;
                    //assert(false);
                }
            }
        }
        std::cout << "numCTGF: " << numCTGF << ", numCFGT: " << numCFGT << std::endl;
        assert(numCTGF == 0 && numCFGT == 0);
        std::cout << "All results are correct ^____^" << std::endl;
    }

    // ----------------
    // Calculate pass rate
    {
        int numPass = 0;
        for (int reqIdx = 0; reqIdx < kNumReqs; reqIdx++)
        {
            for (int docIdx = 0; docIdx < kNumDocs; docIdx++)
            {
                if (rst2D.at(reqIdx).at(docIdx) != 0)
                {
                    numPass++;
                }
            }
        }
        float passRate = (float)numPass / (kNumReqs * kNumDocs);
        std::cout << "numPass: " << numPass << ", pass rate: " << passRate << std::endl;
    }
}

int main()
{
    test3a();
    return 0;
}