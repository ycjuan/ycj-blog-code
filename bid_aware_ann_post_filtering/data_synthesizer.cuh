#ifndef DATA_SYNTHESIZER_CUH
#define DATA_SYNTHESIZER_CUH

#include "data_struct.cuh"

#include <vector>
#include <random>

std::vector<CentroidCpu> genRandCentroids(int numCentroids, int dim, float stdDev)
{
    using namespace std;

    default_random_engine generator;
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

std::vector<ItemCpu> genRandReqDocsFromOneCentroid(
    const CentroidCpu &centroid, float stdDev, int numDocs, float bidStdDev, int &uid)
{
    using namespace std;

    default_random_engine generator;
    normal_distribution<float> embDist(0.0, stdDev);
    normal_distribution<float> bidDist(1.0, bidStdDev);
    uniform_real_distribution<float> attrDist(0.0, 1.0);

    vector<ItemCpu> docs(numDocs);
    for (int i = 0; i < numDocs; i++)
    {
        docs[i].uid = uid++;
        docs[i].centroidId = centroid.centroidId;
        docs[i].emb.resize(centroid.emb.size());
        docs[i].randAttr = attrDist(generator);
        docs[i].bid = max(0.1, bidDist(generator));

        auto &docEmb = docs[i].emb;
        for (int j = 0; j < centroid.emb.size(); j++)
        {
            docEmb[j] = centroid.emb[j] + embDist(generator);
        }
    }

    return docs;
}

std::vector<ItemCpu> genRandReqDocsFromCentroids(
    const std::vector<CentroidCpu>& centroids, float stdDev, int numDocsPerCentroid, float bidStdDev)
{
    using namespace std;

    vector<ItemCpu> docs;
    int uid = 0;
    for (const auto& centroid : centroids)
    {
        auto docsFromOneCentroid = genRandReqDocsFromOneCentroid(centroid, stdDev, numDocsPerCentroid, bidStdDev, uid);
        docs.insert(docs.end(), docsFromOneCentroid.begin(), docsFromOneCentroid.end());
    }

    return docs;
}

#endif // DATA_SYNTHESIZER_CUH