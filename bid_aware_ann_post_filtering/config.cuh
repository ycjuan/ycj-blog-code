#ifndef CONFIG_CUH
#define CONFIG_CUH

const int kNumCentroids = 1000;
const int kDim = 4;
const float kCentroidStdDev = 0.1;
const float kDocStdDev = 0.05;
const float kPassRate = 0.25;
const int kAnnNumtoRetrieveMultiplier = 200;
const int kBidStdDev = 1;
const int kNumDocsPerCentroid = 16000;
const int kNumReqsPerCentroid = 1;
const int kNumToRetrieve = 1000;
const clock_t kFilterSlowdownCycle = 2 << 15;
const clock_t kScoreSlowdownCycle = 2 << 15;
const bool kRunCpu = false;

#endif // CONFIG_CUH