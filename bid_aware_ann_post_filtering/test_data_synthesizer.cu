#include "data_synthesizer.cuh"

#include <iostream>
#include <cassert>

using namespace std;

int testEmb()
{
    int numCentroids = 10;
    int dim = 4;
    float centroidStdDev = 0.1;
    float docStdDev = 0.05;
    float bidStdDev = 1;
    int numDocsPerCentroid = 100;

    vector<CentroidCpu> centroids = genRandCentroids(numCentroids, dim, centroidStdDev);

    vector<ItemCpu> docs = genRandReqDocsFromCentroids(centroids, docStdDev, numDocsPerCentroid, bidStdDev);
    assert(docs.size() == numCentroids * numDocsPerCentroid);

    vector<vector<float>> distance2D(numCentroids, vector<float>(numCentroids, 0.0f));
    vector<vector<int>> count2D(numCentroids, vector<int>(numCentroids, 0));

    double randAttrSum = 0.0;
    int randAttrCount = 0;
    int numBidGt2 = 0;
    for (int d1 = 0; d1 < docs.size(); d1++)
    {
        randAttrSum += docs[d1].randAttr;
        randAttrCount++;
        if (docs[d1].bid > 2.0)
        {
            numBidGt2++;
        }
        for (int d2 = 0; d2 < docs.size(); d2++)
        {
            float dist = 0.0f;
            for (int j = 0; j < dim; j++)
            {
                dist += (docs[d1].emb[j] - docs[d2].emb[j]) * (docs[d1].emb[j] - docs[d2].emb[j]);
            }
            dist = sqrt(dist);
            distance2D[docs[d1].centroidId][docs[d2].centroidId] += dist;
            count2D[docs[d1].centroidId][docs[d2].centroidId]++;
        }
    }
    double randAttrAvg = randAttrSum / randAttrCount;

    double sameCentroidDistSum = 0.0;
    int sameCentroidCount = 0;
    double diffCentroidDistSum = 0.0;
    int diffCentroidCount = 0;
    for (int c1 = 0; c1 < numCentroids; c1++)
    {
        for (int c2 = 0; c2 < numCentroids; c2++)
        {
            distance2D[c1][c2] /= count2D[c1][c2];
            //cout << "distance2D[" << c1 << "][" << c2 << "] = " << distance2D[c1][c2] << endl;
            if (c1 == c2)
            {
                sameCentroidDistSum += distance2D[c1][c2];
                sameCentroidCount++;
            }
            else
            {
                diffCentroidDistSum += distance2D[c1][c2];
                diffCentroidCount++;
            }
        }
    }
    assert(sameCentroidCount == numCentroids);
    assert(diffCentroidCount == numCentroids * (numCentroids - 1));
    double sameCentroidDistAvg = sameCentroidDistSum / sameCentroidCount;
    double diffCentroidDistAvg = diffCentroidDistSum / diffCentroidCount;
    cout << "=====================" << endl;
    cout << "randAttrAvg = " << randAttrAvg << endl;
    cout << "sameCentroidDistAvg = " << sameCentroidDistAvg << endl;
    cout << "diffCentroidDistAvg = " << diffCentroidDistAvg << endl;
    cout << "numBidGt2 = " << numBidGt2 << endl;

    return 0;
}

int main()
{
    testEmb();

    return 0;
}