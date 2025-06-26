#include <chrono>

class Timer
{
public:

    void tic()
    {
        start_ = std::chrono::high_resolution_clock::now();
    }

    long tocMicroSec()
    {
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start_);
        return duration.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};
