#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <unordered_map>

using namespace std;

const int kQPS = 200;
const int kExpTimeSec = 10;
const int kNumReqs = kQPS * kExpTimeSec;

const unordered_map<string, float> kLatencyMsMap =
{
    {"r=AMER_e=0", 10},
    {"r=AMER_e=1", 12},
    {"r=AMER_e=2", 14},
    {"r=EMEA_e=0", 5},
    {"r=EMEA_e=1", 7},
    {"r=EMEA_e=2", 9},
    {"r=APAC_e=0", 1},
    {"r=APAC_e=1", 2},
    {"r=APAC_e=2", 3}
};

string getGroupName(int pct)
{
    if (pct < 0 || pct > 99)
    {
        throw invalid_argument("pct must be between 0 and 99");
    }

    if (pct < 20)
    {
        return "r=AMER_e=0";
    }
    else if (pct < 40)
    {
        return "r=AMER_e=1";
    }
    else if (pct < 50)
    {
        return "r=AMER_e=2";
    }
    else if (pct < 65)
    {
        return "r=EMEA_e=0";
    }
    else if (pct < 75)
    {
        return "r=EMEA_e=1";
    }
    else if (pct < 80)
    {
        return "r=EMEA_e=2";
    }
    else if (pct < 90)
    {
        return "r=APAC_e=0";
    }
    else if (pct < 95)
    {
        return "r=APAC_e=1";
    }
    else
    {
        return "r=APAC_e=2";
    }
}

// Usually, processing multiple requests in a batch is more efficient than processing them one by one.
// For example, if processing 1 request takes 10ms, and processing 2 requests takes 11ms, then the QPS is 2 / (11 / 10) = 1.82x.
const unordered_map<int, float> kBatchLatencyMultiplierMap =
{
    {1, 1.00}, // QPS 1x
    {2, 1.10}, // QPS 2 / 1.10 = 1.82x
    {3, 1.21}, // QPS 3 / 1.21 = 2.48x
    {4, 1.33}, // QPS 4 / 1.33 = 3x
    {5, 1.46}, // QPS 5 / 1.46 = 3.42x
    {6, 1.61}, // QPS 6 / 1.61 = 3.72x
    {7, 1.77}, // QPS 7 / 1.77 = 3.95x
    {8, 1.95}  // QPS 8 / 1.95 = 4.10x
};

#endif