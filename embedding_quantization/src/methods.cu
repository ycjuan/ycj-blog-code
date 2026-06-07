#include "data.cuh"
#include "methods.cuh"
#include "methods_baseline.cuh"
#include "methods_res_quant.cuh"
#include "methods_turbo_quant.cuh"
#include "util.cuh"
#include <vector>

void methodReference(Data data)
{
    std::vector<EMB_T> v_rst(data.config.numToScore * data.config.embDim);
    for (int toBeScoredIdx = 0; toBeScoredIdx < data.config.numToScore; toBeScoredIdx++)
    {
        auto docIdx = data.h_docIdxToScore[toBeScoredIdx];
        for (int embIdx = 0; embIdx < data.config.embDim; embIdx++)
        {
            auto srcMemAddr   = getMemAddr(docIdx, embIdx, data.config.numDocs, data.config.embDim);
            auto dstMemAddr   = getMemAddr(toBeScoredIdx, embIdx, data.config.numToScore, data.config.embDim);
            v_rst[dstMemAddr] = data.h_emb[srcMemAddr];
        }
    }

    CHECK_CUDA(cudaMemcpy(data.d_rst,
                          v_rst.data(),
                          data.config.numToScore * data.config.embDim * sizeof(EMB_T),
                          cudaMemcpyHostToDevice));
}

void runMethod(Data data, Method method)
{
    switch (method)
    {
    case Method::REFERENCE:
        methodReference(data);
        break;
    case Method::BASELINE_H2D:
        methodBaseline(data, true);
        break;
    case Method::BASELINE_D2D:
        methodBaseline(data, false);
        break;
    case Method::RES_QUANT_H2D:
        methodResQuant(data, true);
        break;
    case Method::RES_QUANT_D2D:
        methodResQuant(data, false);
        break;
    case Method::TURBO_QUANT_H2D:
        methodTurboQuant(data, true);
        break;
    case Method::TURBO_QUANT_D2D:
        methodTurboQuant(data, false);
        break;
    default:
        throw std::runtime_error("Invalid method");
    }
}
