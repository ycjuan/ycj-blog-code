#ifndef CONFIG_CUH
#define CONFIG_CUH

const int kNumCentroids = 2000;
const int kDim = 64;
const float kCentroidStdDev = 1;
const float kDocStdDev = 0.5;
const float kPassRate = 0.25;
const float kAnnNumtoRetrieveRatio = 0.05; // retrieve how many % of the total number of documents
const float kBidStdDev = 1.0;
const int kNumDocsPerCentroid = 2000;
const int kNumReqsPerCentroid = 1;
const int kNumToRetrieve = 1000;
const clock_t kFilterSlowdownCycle = 1 << 0;
const clock_t kScoreSlowdownCycle = 1 << 0;
const bool kRunCpu = false;

#endif // CONFIG_CUH
