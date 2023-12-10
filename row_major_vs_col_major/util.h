#include <chrono>

class Timer {

public:
    Timer() {
        reset();
    }

    void reset() {
        tic();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(begin-begin);
    }

    void tic() {
        begin = std::chrono::high_resolution_clock::now();
    }

    void toc() {
        duration += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()-begin);
    }

    double getus() { // microseconds
        return (double)duration.count();
    }

    double getms() { // milliseconds
        return getus() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point begin;
    std::chrono::microseconds duration;
};
