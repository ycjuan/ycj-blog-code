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
    for (int centroidIdx = 0; centroidIdx < config.numCentroids; centroidIdx++)
    {
        for (int embIdx = 0; embIdx < config.embDim; embIdx++)
        {
            const auto addr          = getMemAddr(centroidIdx, embIdx, config.numCentroids, config.embDim);
            data.h_centroidVal[addr] = (EMB_T)distribution(generator);
        }
    }
}

// ------------
// Generate RHT signs for turbo_quant
{
    std::cout << "Generating RHT signs" << std::endl;
    std::default_random_engine         rhtRng(42);
    std::uniform_int_distribution<int> signDist(0, 1);
    for (int i = 0; i < config.embDim; i++)
        data.h_rhtSigns[i] = (signDist(rhtRng) == 0) ? -1 : 1;
}

// ------------
// Generate document embeddings
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

        // Collect all residuals for this document so we can apply RHT afterward
        std::vector<float> residuals(config.embDim);

        for (int embIdx = 0; embIdx < config.embDim; embIdx++)
        {
            auto  centroid = data.h_centroidVal[getMemAddr(centroidIdx, embIdx, config.numCentroids, config.embDim)];
            float residual = norm_distributions[omp_get_thread_num()](generators[omp_get_thread_num()]);
            float emb      = static_cast<float>(centroid) + residual;
            data.h_emb[getMemAddr(docIdx, embIdx, config.numDocs, config.embDim)] = static_cast<EMB_T>(emb);
            auto rqIdx     = getRqIdx(embIdx, config.numBitsPerDim, kBitsPerInt);
            auto rqMemAddr = getMemAddr(docIdx, rqIdx, config.numDocs, config.getRqDim());
            quantize(config.numBitsPerDim, kBitsPerInt, config.stdDev, residual, data.h_residual[rqMemAddr], embIdx);
            residuals[embIdx] = residual;
        }

        // Apply RHT to residuals: x_rot = (1/sqrt(d)) * WHT(signs * x)
        // Step 1: sign flip
        for (int i = 0; i < config.embDim; i++)
            residuals[i] *= data.h_rhtSigns[i];
        // Step 2: Walsh-Hadamard Transform (in-place)
        for (int stride = 1; stride < config.embDim; stride <<= 1)
            for (int i = 0; i < config.embDim; i += 2 * stride)
                for (int j = 0; j < stride; j++)
                {
                    float a                   = residuals[i + j];
                    float b                   = residuals[i + j + stride];
                    residuals[i + j]          = a + b;
                    residuals[i + j + stride] = a - b;
                }
        // Step 3: normalize and Lloyd-Max quantize
        float scale = 1.0f / sqrtf((float)config.embDim);
        for (int embIdx = 0; embIdx < config.embDim; embIdx++)
        {
            float rotated   = residuals[embIdx] * scale;
            auto  rqIdx     = getRqIdx(embIdx, config.numBitsPerDim, kBitsPerInt);
            auto  rqMemAddr = getMemAddr(docIdx, rqIdx, config.numDocs, config.getRqDim());
            lloydMaxQuantize(config.numBitsPerDim,
                             kBitsPerInt,
                             config.stdDev,
                             rotated,
                             data.h_turboRes[rqMemAddr],
                             embIdx);
        }
    }
}

// ------------
// Generate document indices to score
{
    std::cout << "Generating document indices to score" << std::endl;
    std::vector<int> docIdxToScore(config.numDocs);
    for (int i = 0; i < config.numDocs; i++)
        docIdxToScore[i] = i;
    std::default_random_engine generator;
    std::shuffle(docIdxToScore.begin(), docIdxToScore.end(), generator);
    docIdxToScore.resize(config.numToScore);
    std::sort(docIdxToScore.begin(), docIdxToScore.end());
    for (int i = 0; i < config.numToScore; i++)
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
