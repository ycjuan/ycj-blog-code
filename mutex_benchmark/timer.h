#ifndef TIMER_CUH
#define TIMER_CUH

class Timer
{
public:

    void tic()
    {
        start_ = std::chrono::high_resolution_clock::now();
    }

    int64_t tocNs()
    {
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::nanoseconds duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start_);
        return duration.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

#endif