#ifndef DATA_SYNTHESIZER_CUH
#define DATA_SYNTHESIZER_CUH

#include "data_struct.cuh"

#include <vector>
#include <random>

std::vector<CentroidCpu> genRandCentroids(int numCentroids, int dim, float stdDev)
{
    using namespace std;

    default_random_engine generator;
    generator.seed(0);
    normal_distribution<float> distribution(0.0, stdDev);

    vector<CentroidCpu> centroids(numCentroids);
    for (int i = 0; i < numCentroids; i++)
    {
        centroids[i].centroidId = i;
        centroids[i].emb.resize(dim);
        auto& centroidEmb = centroids[i].emb;
        for (int j = 0; j < dim; j++)
        {
            centroidEmb[j] = distribution(generator);
        }
    }

    return centroids;
}

std::vector<ItemCpu> genRandItemsFromOneCentroid(
    const CentroidCpu &centroid, float stdDev, int numItems, float bidStdDev, int &uid)
{
    using namespace std;

    default_random_engine generator;
    generator.seed(uid);
    normal_distribution<float> embDist(0.0, stdDev);
    normal_distribution<float> bidDist(1.0, bidStdDev);
    uniform_real_distribution<float> attrDist(0.0, 1.0);

    vector<ItemCpu> items(numItems);
    for (int i = 0; i < numItems; i++)
    {
        items[i].uid = uid++;
        items[i].centroidId = centroid.centroidId;
        items[i].emb.resize(centroid.emb.size());
        items[i].randAttr = attrDist(generator);
        items[i].bid = max(0.1, bidDist(generator));

        auto &docEmb = items[i].emb;
        for (int j = 0; j < centroid.emb.size(); j++)
        {
            docEmb[j] = centroid.emb[j] + embDist(generator);
        }
    }

    return items;
}

std::vector<ItemCpu> genRandReqDocsFromCentroids(
    const std::vector<CentroidCpu>& centroids, float stdDev, int numItemsPerCentroid, float bidStdDev)
{
    using namespace std;

    vector<ItemCpu> items;
    int uid = 0;
    for (const auto& centroid : centroids)
    {
        auto itemsFromOneCentroid = genRandItemsFromOneCentroid(centroid, stdDev, numItemsPerCentroid, bidStdDev, uid);
        items.insert(items.end(), itemsFromOneCentroid.begin(), itemsFromOneCentroid.end());
    }

    return items;
}

#endif // DATA_SYNTHESIZER_CUH