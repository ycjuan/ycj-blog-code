#include "util.cuh"

#include <unistd.h>
#include <ios>
#include <fstream>


CudaTimer::CudaTimer()
{
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
}

void CudaTimer::tic()
{
    cudaEventRecord(start_);
}

float CudaTimer::tocMs()
{
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    float elapsedMs;
    cudaEventElapsedTime(&elapsedMs, start_, stop_);
    return elapsedMs;
}

CudaTimer::~CudaTimer()
{
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

/*
Reference:
- https://stackoverflow.com/questions/669438/how-to-get-memory-usage-at-runtime-using-c
- https://docs.hpc.qmul.ac.uk/using/memory/#:~:text=Memory%20usage%20can%20be%20broadly,amount%20of%20memory%20it%20uses
- https://www.baeldung.com/linux/resident-set-vs-virtual-memory-size

*/
size_t getCpuRamUsageByte()
{
    using std::ifstream;
    using std::ios_base;
    using std::string;

    // 'file' stat seems to give the most reliable results
    //
    ifstream stat_stream("/proc/self/stat", ios_base::in);

    // dummy vars for leading entries in stat that we don't care about
    //
    string pid, comm, state, ppid, pgrp, session, tty_nr;
    string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    string utime, stime, cutime, cstime, priority, nice;
    string O, itrealvalue, starttime;

    // the two fields we want
    //
    unsigned long vsize;
    long rss;

    stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt >> utime >> stime >> cutime >> cstime >> priority >> nice >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

    stat_stream.close();

    long page_size_byte = sysconf(_SC_PAGE_SIZE); // in case x86-64 is configured to use 2MB pages
    return rss * page_size_byte;
}

float getCpuRamUsageMiB()
{
    return getCpuRamUsageByte() / (1024.0 * 1024.0);
}