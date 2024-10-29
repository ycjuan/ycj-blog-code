#include "data_synthesizer.cuh"
#include "config.cuh"

#include <iostream>
#include <cassert>


using namespace std;

int main()
{
    vector<CentroidCpu> centroids = genRandCentroids(kNumCentroids, kDim, kCentroidStdDev);

    vector<ItemCpu> docs = genRandReqDocsFromCentroids(centroids, kDocStdDev, kNumDocsPerCentroid, kBidStdDev);
    assert(docs.size() == kNumCentroids * kNumDocsPerCentroid);

    vector<vector<float>> distance2D(kNumCentroids, vector<float>(kNumCentroids, 0.0f));
    vector<vector<int>> count2D(kNumCentroids, vector<int>(kNumCentroids, 0));

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
            for (int j = 0; j < kDim; j++)
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
    for (int c1 = 0; c1 < kNumCentroids; c1++)
    {
        for (int c2 = 0; c2 < kNumCentroids; c2++)
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
    assert(sameCentroidCount == kNumCentroids);
    assert(diffCentroidCount == kNumCentroids * (kNumCentroids - 1));
    double sameCentroidDistAvg = sameCentroidDistSum / sameCentroidCount;
    double diffCentroidDistAvg = diffCentroidDistSum / diffCentroidCount;
    double bidGt2Ratio = (double)numBidGt2 / docs.size();
    cout << "=====================" << endl;
    cout << "randAttrAvg = " << randAttrAvg << endl;
    cout << "sameCentroidDistAvg = " << sameCentroidDistAvg << endl;
    cout << "diffCentroidDistAvg = " << diffCentroidDistAvg << endl;
    cout << "bidGt2Ratio = " << bidGt2Ratio << endl;

    return 0;
}