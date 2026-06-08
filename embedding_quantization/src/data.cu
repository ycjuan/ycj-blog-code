#include "data.cuh"
#include "util.cuh"
#include <algorithm>
#include <cuda_runtime.h>
#include <omp.h>
#include <random>

Data genData(Config config)
{
    Data data;
    data.config = config;

    // ----------------
    // Allocate memory
    {
        std::cout << "Allocating memory for data" << std::endl;
        CHECK_CUDA(cudaMallocHost(&data.h_emb, config.numDocs * config.embDim * sizeof(EMB_T)));
        CHECK_CUDA(cudaMalloc(&data.d_emb, config.numDocs * config.embDim * sizeof(EMB_T)));
        CHECK_CUDA(cudaMallocHost(&data.h_centroidVal, config.numCentroids * config.embDim * sizeof(EMB_T)));
        CHECK_CUDA(cudaMalloc(&data.d_centroidVal, config.numCentroids * config.embDim * sizeof(EMB_T)));
        CHECK_CUDA(cudaMallocHost(&data.h_centroidIdx, config.numDocs * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&data.d_centroidIdx, config.numDocs * sizeof(int)));
        CHECK_CUDA(cudaMallocHost(&data.h_residual, config.numDocs * config.getRqDim() * sizeof(RQ_T)));
        memset(data.h_residual, 0, config.numDocs * config.getRqDim() * sizeof(RQ_T));
        CHECK_CUDA(cudaMalloc(&data.d_residual, config.numDocs * config.getRqDim() * sizeof(RQ_T)));
        CHECK_CUDA(cudaMallocHost(&data.h_centroidStd, config.numCentroids * config.embDim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&data.d_centroidStd, config.numCentroids * config.embDim * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&data.h_centroidEffStd, config.numCentroids * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&data.d_centroidEffStd, config.numCentroids * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&data.h_rhtSigns, config.embDim * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&data.d_rhtSigns, config.embDim * sizeof(int)));
        CHECK_CUDA(cudaMallocHost(&data.h_turboRes, config.numDocs * config.getRqDim() * sizeof(RQ_T)));
        memset(data.h_turboRes, 0, config.numDocs * config.getRqDim() * sizeof(RQ_T));
        CHECK_CUDA(cudaMalloc(&data.d_turboRes, config.numDocs * config.getRqDim() * sizeof(RQ_T)));
        CHECK_CUDA(cudaMallocHost(&data.h_docIdxToScore, config.numToScore * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&data.d_docIdxToScore, config.numToScore * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&data.d_rst, config.numToScore * config.embDim * sizeof(EMB_T)));
    }

    // ----------------
    // Generate data
    { // ------------
      // Generate centroid embeddings
      { std::cout << "Generating centroid embeddings" << std::endl;
    std::default_random_engine            generator;
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);
    for (int centroidIdx = 0; centroidIdx < (int)config.numCentroids; centroidIdx++)
        for (int embIdx = 0; embIdx < (int)config.embDim; embIdx++)
            data.h_centroidVal[getMemAddr(centroidIdx, embIdx, config.numCentroids, config.embDim)]
                = (EMB_T)distribution(generator);
}

// ------------
// Generate RHT signs for turbo_quant
{
    std::cout << "Generating RHT signs" << std::endl;
    std::default_random_engine         rhtRng(42);
    std::uniform_int_distribution<int> signDist(0, 1);
    for (int i = 0; i < (int)config.embDim; i++)
        data.h_rhtSigns[i] = (signDist(rhtRng) == 0) ? -1 : 1;
}

// ------------
// Phase 1: generate document embeddings (no quantization yet)
{
    std::cout << "Generating document embeddings" << std::endl;
    int                                          numThreads = omp_get_max_threads();
    std::vector<std::default_random_engine>      generators(numThreads);
    std::vector<std::normal_distribution<float>> norm_distributions(numThreads);
    for (int t = 0; t < numThreads; t++)
    {
        generators[t]         = std::default_random_engine(t);
        norm_distributions[t] = std::normal_distribution<float>(0.0, config.stdDev);
    }

#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (size_t docIdx = 0; docIdx < config.numDocs; docIdx++)
    {
        int centroidIdx            = docIdx % config.numCentroids;
        data.h_centroidIdx[docIdx] = centroidIdx;
        int tid                    = omp_get_thread_num();
        for (int embIdx = 0; embIdx < (int)config.embDim; embIdx++)
        {
            float centroid = static_cast<float>(
                data.h_centroidVal[getMemAddr(centroidIdx, embIdx, config.numCentroids, config.embDim)]);
            float residual = norm_distributions[tid](generators[tid]);
            data.h_emb[getMemAddr(docIdx, embIdx, config.numDocs, config.embDim)]
                = static_cast<EMB_T>(centroid + residual);
        }
    }
}

// ------------
// Phase 2: compute per-dim per-centroid stdDev and per-centroid effective stdDev
// centroidIdx = docIdx % numCentroids, so each centroid's docs are perfectly strided —
// parallelize over centroids with no locking needed.
{
    std::cout << "Computing per-centroid stdDev" << std::endl;
#pragma omp parallel for schedule(dynamic)
    for (int centroidIdx = 0; centroidIdx < (int)config.numCentroids; centroidIdx++)
    {
        int                 cnt = 0;
        std::vector<double> sumSq(config.embDim, 0.0);
        for (size_t docIdx = centroidIdx; docIdx < config.numDocs; docIdx += config.numCentroids)
        {
            cnt++;
            for (int embIdx = 0; embIdx < (int)config.embDim; embIdx++)
            {
                float emb = static_cast<float>(data.h_emb[getMemAddr(docIdx, embIdx, config.numDocs, config.embDim)]);
                float centroid = static_cast<float>(
                    data.h_centroidVal[getMemAddr(centroidIdx, embIdx, config.numCentroids, config.embDim)]);
                double r = emb - centroid;
                sumSq[embIdx] += r * r;
            }
        }
        double effVarSum = 0.0;
        for (int embIdx = 0; embIdx < (int)config.embDim; embIdx++)
        {
            double var = sumSq[embIdx] / cnt;
            data.h_centroidStd[getMemAddr(centroidIdx, embIdx, config.numCentroids, config.embDim)]
                = static_cast<float>(std::sqrt(var));
            effVarSum += var;
        }
        // Effective stdDev for TurboQuant: RMS of per-dim stdDevs.
        // After RHT, rotated residuals have variance = mean of per-dim variances.
        data.h_centroidEffStd[centroidIdx] = static_cast<float>(std::sqrt(effVarSum / config.embDim));
    }
}

// ------------
// Phase 3: quantize residuals using computed stdDevs
{
    std::cout << "Quantizing residuals" << std::endl;
    int numThreads = omp_get_max_threads();
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (size_t docIdx = 0; docIdx < config.numDocs; docIdx++)
    {
        int   centroidIdx = data.h_centroidIdx[docIdx];
        float effStd      = data.h_centroidEffStd[centroidIdx];

        std::vector<float> residuals(config.embDim);
        for (int embIdx = 0; embIdx < (int)config.embDim; embIdx++)
        {
            float emb      = static_cast<float>(data.h_emb[getMemAddr(docIdx, embIdx, config.numDocs, config.embDim)]);
            float centroid = static_cast<float>(
                data.h_centroidVal[getMemAddr(centroidIdx, embIdx, config.numCentroids, config.embDim)]);
            float residual    = emb - centroid;
            residuals[embIdx] = residual;

            // ResQuant: per-dim per-centroid stdDev
            float std       = data.h_centroidStd[getMemAddr(centroidIdx, embIdx, config.numCentroids, config.embDim)];
            auto  rqIdx     = getRqIdx(embIdx, config.numBitsPerDim, kBitsPerInt);
            auto  rqMemAddr = getMemAddr(docIdx, rqIdx, config.numDocs, config.getRqDim());
            quantize(config.numBitsPerDim, kBitsPerInt, std, residual, data.h_residual[rqMemAddr], embIdx);
        }

        // TurboQuant: RHT then lloydMaxQuantize with per-centroid effective stdDev
        for (int i = 0; i < (int)config.embDim; i++)
            residuals[i] *= data.h_rhtSigns[i];
        for (int stride = 1; stride < (int)config.embDim; stride <<= 1)
            for (int i = 0; i < (int)config.embDim; i += 2 * stride)
                for (int j = 0; j < stride; j++)
                {
                    float a                   = residuals[i + j];
                    float b                   = residuals[i + j + stride];
                    residuals[i + j]          = a + b;
                    residuals[i + j + stride] = a - b;
                }
        float scale = 1.0f / sqrtf((float)config.embDim);
        for (int embIdx = 0; embIdx < (int)config.embDim; embIdx++)
        {
            float rotated   = residuals[embIdx] * scale;
            auto  rqIdx     = getRqIdx(embIdx, config.numBitsPerDim, kBitsPerInt);
            auto  rqMemAddr = getMemAddr(docIdx, rqIdx, config.numDocs, config.getRqDim());
            lloydMaxQuantize(config.numBitsPerDim, kBitsPerInt, effStd, rotated, data.h_turboRes[rqMemAddr], embIdx);
        }
    }
}

// ------------
// Generate document indices to score
{
    std::cout << "Generating document indices to score" << std::endl;
    std::vector<int> docIdxToScore(config.numDocs);
    for (int i = 0; i < (int)config.numDocs; i++)
        docIdxToScore[i] = i;
    std::default_random_engine generator;
    std::shuffle(docIdxToScore.begin(), docIdxToScore.end(), generator);
    docIdxToScore.resize(config.numToScore);
    std::sort(docIdxToScore.begin(), docIdxToScore.end());
    for (int i = 0; i < (int)config.numToScore; i++)
        data.h_docIdxToScore[i] = docIdxToScore[i];
}
}

// ----------------
// Copy data to device
{
    std::cout << "Copying data to device" << std::endl;
    CHECK_CUDA(
        cudaMemcpy(data.d_emb, data.h_emb, config.numDocs * config.embDim * sizeof(EMB_T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(data.d_centroidVal,
                          data.h_centroidVal,
                          config.numCentroids * config.embDim * sizeof(EMB_T),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(data.d_centroidIdx, data.h_centroidIdx, config.numDocs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(data.d_residual,
                          data.h_residual,
                          config.numDocs * config.getRqDim() * sizeof(RQ_T),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(data.d_centroidStd,
                          data.h_centroidStd,
                          config.numCentroids * config.embDim * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(data.d_centroidEffStd,
                          data.h_centroidEffStd,
                          config.numCentroids * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(data.d_rhtSigns, data.h_rhtSigns, config.embDim * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(data.d_turboRes,
                          data.h_turboRes,
                          config.numDocs * config.getRqDim() * sizeof(RQ_T),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(data.d_docIdxToScore,
                          data.h_docIdxToScore,
                          config.numToScore * sizeof(int),
                          cudaMemcpyHostToDevice));
}

return data;
}
