#include "core.h"
#include "util.h"

#include <iostream>

using namespace std;

int main(int argc, char* argv[])
{
    int numTrials = 2000;
    int numThreads = 16;

    cout << "numThreads: " << numThreads << ", numTrials: " << numTrials << endl;

    // Create an instance of Core
    Core core;

    // Call the process method with the specified number of threads
    Timer timer;
    for (int i = -3; i < numTrials; ++i) 
    {
        if (i == 0) 
        {
            timer.tic();
        }
        core.process(numThreads);
    }
    double avgTimeMs = timer.tocMicroSec() / (double)numTrials / 1000.0;
    std::cout << "Average time per trial: " << avgTimeMs << " ms" << std::endl;

    return 0;
}