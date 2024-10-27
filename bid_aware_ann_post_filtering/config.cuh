#ifndef CONFIG_CUH
#define CONFIG_CUH

const int kNumCentroids = 100;
const int kDim = 4;
const float kCentroidStdDev = 0.1;
const float kDocStdDev = 0.05;
const float kPassRate = 0.25;
const float kMinPassRate = 0.01;
const int kBidStdDev = 1;
const int kNumDocsPerCentroid = 10000;
const int kNumReqsPerCentroid = 1;
const int kNumToRetrieve = 1000;

#endif // CONFIG_CUH