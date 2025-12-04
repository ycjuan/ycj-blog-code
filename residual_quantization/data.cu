#include "data.cuh"
#include "util.cuh"
#include <omp.h>
#include <cuda_runtime.h>
#include <random>
#include <algorithm>

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
        CHECK_CUDA(cudaMallocHost(&data.h_centroidEmb, config.numCentroids * config.embDim * 2 * sizeof(EMB_T)));
        CHECK_CUDA(cudaMalloc(&data.d_centroidEmb, config.numCentroids * config.embDim * 2 * sizeof(EMB_T)));
        CHECK_CUDA(cudaMallocHost(&data.h_centroidIdx, config.numDocs * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&data.d_centroidIdx, config.numDocs * sizeof(int)));
        CHECK_CUDA(cudaMallocHost(&data.h_residual, config.numDocs * config.getRqDim() * sizeof(RQ_T)));
        CHECK_CUDA(cudaMallocHost(&data.h_docIdxToScore, config.numToScore * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&data.d_docIdxToScore, config.numToScore * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&data.d_residual, config.numDocs * config.getRqDim() * sizeof(RQ_T)));
        CHECK_CUDA(cudaMalloc(&data.d_rst, config.numToScore * config.embDim * sizeof(EMB_T)));    
    }

    // ----------------
    // Generate data
    {
        // ------------
        // Generate centroid embeddings
        {
            std::cout << "Generating centroid embeddings" << std::endl;
            std::default_random_engine generator;
            std::uniform_real_distribution<float> distribution(-1.0, 1.0);
            for (int centroidIdx = 0; centroidIdx < config.numCentroids; centroidIdx++) 
            {
                for (int embIdx = 0; embIdx < config.embDim; embIdx++)
                {
                    const auto addr = getMemAddr(centroidIdx, embIdx * 2, config.numCentroids, config.embDim * 2);                    
                    data.h_centroidEmb[addr] = (EMB_T)distribution(generator);
                    data.h_centroidEmb[addr + 1] = (EMB_T)config.stdDev;
                }
            }
        }

        // ------------
        // Generate document embeddings
        {
            std::cout << "Generating document embeddings" << std::endl;
            int numThreads = omp_get_max_threads();
            std::vector<std::default_random_engine> generators(numThreads);
            std::vector<std::normal_distribution<float>> norm_distributions(numThreads);
            for (int t = 0; t < numThreads; t++)
            {
                generators[t] = std::default_random_engine(t);
                norm_distributions[t] = std::normal_distribution<float>(0.0, config.stdDev);
            }

            #pragma omp parallel for num_threads(numThreads) schedule(static)
            for (size_t docIdx = 0; docIdx < config.numDocs; docIdx++) 
            {
                int centroidIdx = docIdx % config.numCentroids;
                data.h_centroidIdx[docIdx] = centroidIdx;
                for (int embIdx = 0; embIdx < config.embDim; embIdx++)
                {
                    auto centroid = data.h_centroidEmb[getMemAddr(centroidIdx, embIdx * 2, config.numCentroids, config.embDim * 2)];
                    auto residual = (EMB_T)(norm_distributions[omp_get_thread_num()](generators[omp_get_thread_num()]));
                    data.h_emb[getMemAddr(docIdx, embIdx, config.numDocs, config.embDim)] = centroid + residual;
                    auto rqIdx = getRqIdx(embIdx, config.numBitsPerDim, kBitsPerInt);
                    auto rqMemAddr = getMemAddr(docIdx, rqIdx, config.numDocs, config.getRqDim());
                    quantize(config.numBitsPerDim, kBitsPerInt, config.stdDev, residual, data.h_residual[rqMemAddr], embIdx);
                }
            }        
        }

        // ------------
        // Generate document indices to score
        {
            std::cout << "Generating document indices to score" << std::endl;
            std::vector<int> docIdxToScore(config.numToScore);
            for (int i = 0; i < config.numToScore; i++)
            {
                docIdxToScore[i] = i;
            }
            std::default_random_engine generator;
            std::shuffle(docIdxToScore.begin(), docIdxToScore.end(), generator);
            for (int i = 0; i < config.numToScore; i++)
            {
                data.h_docIdxToScore[i] = docIdxToScore[i];
            }
        }
    }

    // ----------------
    // Copy data to device
    {
        std::cout << "Copying data to device" << std::endl;
        CHECK_CUDA(cudaMemcpy(data.d_emb, data.h_emb, config.numDocs * config.embDim * sizeof(EMB_T), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(data.d_centroidEmb, data.h_centroidEmb, config.numCentroids * config.embDim * 2* sizeof(EMB_T), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(data.d_centroidIdx, data.h_centroidIdx, config.numDocs * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(data.d_residual, data.h_residual, config.numDocs * config.getRqDim() * sizeof(RQ_T), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(data.d_docIdxToScore, data.h_docIdxToScore, config.numToScore * sizeof(int), cudaMemcpyHostToDevice));
    }

    return data;
}