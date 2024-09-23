#include <random>
#include <vector>
#include <iostream>

#include "util.h"

const int kNumTrials = 100;
const size_t kVectorSize = 1000000;

using namespace std;

int main()
{
    default_random_engine generator;
    uniform_real_distribution<float> distribution(-1.0, 1.0);

    cout << "kVectorSize = " << kVectorSize << endl;
    cout << "kNumTrials = " << kNumTrials << endl;

    vector<float> vec(kVectorSize);
    for (int i = 0; i < kVectorSize; i++)
        vec[i] = distribution(generator);

    Timer timer;
    timer.tic();
    double sum = 0;
    for (int t = 0; t < kNumTrials; t++)
        for (int i = 0; i < kVectorSize; i++)
            sum += vec[i];
    cout << "sum = " << sum << endl;
    cout << "time using [] = " << timer.tocMs() << " ms" << endl;

    timer.tic();
    sum = 0;
    for (int t = 0; t < kNumTrials; t++)
        for (int i = 0; i < kVectorSize; i++)
            sum += vec.at(i);
    cout << "sum = " << sum << endl;
    cout << "time using .at() = " << timer.tocMs() << " ms" << endl;

    return 0;
}
