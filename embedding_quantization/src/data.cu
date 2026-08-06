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
// Phase 2: compute per-dim per-centroid stdDev (for res_quant) and global turboQuantStdDev.
// centroidIdx = docIdx % numCentroids, so strided iteration over docs is race-free per centroid.
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
        for (int embIdx = 0; embIdx < (int)config.embDim; embIdx++)
            data.h_centroidStd[getMemAddr(centroidIdx, embIdx, config.numCentroids, config.embDim)]
                = static_cast<float>(std::sqrt(sumSq[embIdx] / cnt));
    }

    // turboQuantStdDev = RMS of raw embedding values across all docs and dims.
    // Computed cheaply from centroid means and per-dim residual variances:
    //   E[emb²] = E[(centroid + residual)²] = centroid² + residual_var  (cross term is zero)
    // Averaged uniformly across all centroids (valid since numDocs/numCentroids is constant).
    double sumSqTotal = 0.0;
    for (int centroidIdx = 0; centroidIdx < (int)config.numCentroids; centroidIdx++)
        for (int embIdx = 0; embIdx < (int)config.embDim; embIdx++)
        {
            float cv = static_cast<float>(
                data.h_centroidVal[getMemAddr(centroidIdx, embIdx, config.numCentroids, config.embDim)]);
            float cs = data.h_centroidStd[getMemAddr(centroidIdx, embIdx, config.numCentroids, config.embDim)];
            sumSqTotal += (double)cv * cv + (double)cs * cs;
        }
    data.turboQuantStdDev = static_cast<float>(std::sqrt(sumSqTotal / ((double)config.numCentroids * config.embDim)));
}

// ------------
// Phase 3: quantize residuals
{
    std::cout << "Quantizing residuals" << std::endl;
    int numThreads = omp_get_max_threads();
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (size_t docIdx = 0; docIdx < config.numDocs; docIdx++)
    {
        int centroidIdx = data.h_centroidIdx[docIdx];

        // ResQuant: quantize residual (emb - centroid) with per-dim per-centroid stdDev
        for (int embIdx = 0; embIdx < (int)config.embDim; embIdx++)
        {
            float emb      = static_cast<float>(data.h_emb[getMemAddr(docIdx, embIdx, config.numDocs, config.embDim)]);
            float centroid = static_cast<float>(
                data.h_centroidVal[getMemAddr(centroidIdx, embIdx, config.numCentroids, config.embDim)]);
            float residual  = emb - centroid;
            float std       = data.h_centroidStd[getMemAddr(centroidIdx, embIdx, config.numCentroids, config.embDim)];
            auto  rqIdx     = getRqIdx(embIdx, config.numBitsPerDim, kBitsPerInt);
            auto  rqMemAddr = getMemAddr(docIdx, rqIdx, config.numDocs, config.getRqDim());
            quantize(config.numBitsPerDim, kBitsPerInt, std, residual, data.h_residual[rqMemAddr], embIdx);
        }

        // TurboQuant: apply RHT to the raw embedding (no centroid subtraction),
        // then lloydMaxQuantize with the global turboQuantStdDev.
        std::vector<float> embs(config.embDim);
        for (int embIdx = 0; embIdx < (int)config.embDim; embIdx++)
            embs[embIdx] = static_cast<float>(data.h_emb[getMemAddr(docIdx, embIdx, config.numDocs, config.embDim)]);
        for (int i = 0; i < (int)config.embDim; i++)
            embs[i] *= data.h_rhtSigns[i];
        for (int stride = 1; stride < (int)config.embDim; stride <<= 1)
            for (int i = 0; i < (int)config.embDim; i += 2 * stride)
                for (int j = 0; j < stride; j++)
                {
                    float a              = embs[i + j];
                    float b              = embs[i + j + stride];
                    embs[i + j]          = a + b;
                    embs[i + j + stride] = a - b;
                }
        float scale = 1.0f / sqrtf((float)config.embDim);
        for (int embIdx = 0; embIdx < (int)config.embDim; embIdx++)
        {
            float rotated   = embs[embIdx] * scale;
            auto  rqIdx     = getRqIdx(embIdx, config.numBitsPerDim, kBitsPerInt);
            auto  rqMemAddr = getMemAddr(docIdx, rqIdx, config.numDocs, config.getRqDim());
            lloydMaxQuantize(config.numBitsPerDim,
                             kBitsPerInt,
                             data.turboQuantStdDev,
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
