#include <string>
#include <stdexcept>
#include <iostream>
#include <random>
#include <sstream>
#include <cublas_v2.h>
#include <type_traits>

#include "util.cuh"
#include "common.cuh"
#include "methods.cuh"

using namespace std;

size_t kNumRows = 1 << 20;
size_t kNumCols = 1 << 10;
int kNumTrials = 10;

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

void printBandwidth(const string setting, const size_t dataSizeInBytes, const double timeMs)
{
    float bandwidthGiBs = dataSizeInBytes / 1024.0 / 1024.0 / 1024.0 / (timeMs / kNumTrials / 1000.0);
    cout << setting << ": " << bandwidthGiBs << " GiB/s" << endl;
}

int main()
{
    // ----------------------------------------
    // prepare data
    Data data;
    data.malloc(kNumRows, kNumCols);
    data.print();

    // ----------------------------------------
    // h2h
    {
        // r2r - memcpy (pageable)
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                memcpy(data.h_data_dst, data.h_data_src, data.dataSizeInBytes);
            }
            printBandwidth("h2h_r2r memcpy (pageable)", data.dataSizeInBytes, timer.tocMs());
        }

        // r2r - memcpy (pinned)
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                memcpy(data.hp_data_dst, data.hp_data_src, data.dataSizeInBytes);
            }
            printBandwidth("h2h_r2r memcpy (pinned)", data.dataSizeInBytes, timer.tocMs());
        }

        // r2r - cudaMemcpy
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                CHECK_CUDA(cudaMemcpy(data.hp_data_dst, data.hp_data_src, data.dataSizeInBytes, cudaMemcpyHostToHost));
                CHECK_CUDA(cudaDeviceSynchronize());
            }
            printBandwidth("h2h_r2r cudaMemcpy (pinned)", data.dataSizeInBytes, timer.tocMs());
        }

        // r2r - custom
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                h2h_r2r(data);
            }
            printBandwidth("h2h_r2r custom", data.dataSizeInBytes, timer.tocMs());
        }

        // r2c - custom
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                h2h_r2c(data);
            }
            printBandwidth("h2h_r2c custom", data.dataSizeInBytes, timer.tocMs());
        }

        // c2r - custom
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                h2h_c2r(data);
            }
            printBandwidth("h2h_c2r custom", data.dataSizeInBytes, timer.tocMs());
        }

        // c2c - custom
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                h2h_c2c(data);
            }
            printBandwidth("h2h_c2c custom", data.dataSizeInBytes, timer.tocMs());
        }
    }

    // ----------------------------------------
    // h2d
    {
        // r2r - cudaMemcpy (pageable)
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                CHECK_CUDA(cudaMemcpy(data.d_data_dst, data.h_data_src, data.dataSizeInBytes, cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaDeviceSynchronize());
            }
            printBandwidth("h2d_r2r cudaMemcpy (pageable)", data.dataSizeInBytes, timer.tocMs());
        }

        // r2r - cudaMemcpy (pinned)
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                CHECK_CUDA(cudaMemcpy(data.d_data_dst, data.hp_data_src, data.dataSizeInBytes, cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaDeviceSynchronize());
            }
            printBandwidth("h2d_r2r cudaMemcpy (pinned)", data.dataSizeInBytes, timer.tocMs());
        }

        // r2r - custom
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                h2d_r2r(data);
            }
            printBandwidth("h2d_r2r custom", data.dataSizeInBytes, timer.tocMs());
        }

        // r2c - custom
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                h2d_r2c(data);
            }
            printBandwidth("h2d_r2c custom", data.dataSizeInBytes, timer.tocMs());
        }

        // c2r - custom
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                h2d_c2r(data);
            }
            printBandwidth("h2d_c2r custom", data.dataSizeInBytes, timer.tocMs());
        }

        // c2c - custom
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                h2d_c2c(data);
            }
            printBandwidth("h2d_c2c custom", data.dataSizeInBytes, timer.tocMs());
        }
    }

    // ----------------------------------------
    // d2h
    {
        // r2r - cudaMemcpy (pageable)
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                CHECK_CUDA(cudaMemcpy(data.h_data_dst, data.d_data_src, data.dataSizeInBytes, cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaDeviceSynchronize());
            }
            printBandwidth("d2h_r2r cudaMemcpy (pageable)", data.dataSizeInBytes, timer.tocMs());
        }

        // r2r - cudaMemcpy (pinned)
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                CHECK_CUDA(cudaMemcpy(data.hp_data_dst, data.d_data_src, data.dataSizeInBytes, cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaDeviceSynchronize());
            }
            printBandwidth("d2h_r2r cudaMemcpy (pinned)", data.dataSizeInBytes, timer.tocMs());
        }

        // r2r - custom
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                d2h_r2r(data);
            }
            printBandwidth("d2h_r2r custom", data.dataSizeInBytes, timer.tocMs());
        }

        // r2c - custom
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                d2h_r2c(data);
            }
            printBandwidth("d2h_r2c custom", data.dataSizeInBytes, timer.tocMs());
        }

        // c2r - custom
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                d2h_c2r(data);
            }
            printBandwidth("d2h_c2r custom", data.dataSizeInBytes, timer.tocMs());
        }

        // c2c - custom
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                d2h_c2c(data);
            }
            printBandwidth("d2h_c2c custom", data.dataSizeInBytes, timer.tocMs());
        }
    }

    // ----------------------------------------
    // d2d
    {
        // r2r - cudaMemcpy
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                CHECK_CUDA(cudaMemcpy(data.d_data_dst, data.d_data_src, data.dataSizeInBytes, cudaMemcpyDeviceToDevice));
                CHECK_CUDA(cudaDeviceSynchronize());
            }
            printBandwidth("d2d_r2r cudaMemcpy", data.dataSizeInBytes, timer.tocMs());
        }

        // r2r - custom
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                d2d_r2r(data);
            }
            printBandwidth("d2d_r2r custom", data.dataSizeInBytes, timer.tocMs());
        }

        // r2c - custom
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                d2d_r2c(data);
            }
            printBandwidth("d2d_r2c custom", data.dataSizeInBytes, timer.tocMs());
        }

        // c2r - custom
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                d2d_c2r(data);
            }
            printBandwidth("d2d_c2r custom", data.dataSizeInBytes, timer.tocMs());
        }

        // c2c - custom
        {
            Timer timer;
            timer.tic();
            for (int t = -3; t < kNumTrials; t++)
            {
                if (t == 0)
                {
                    timer.tic();
                }
                d2d_c2c(data);
            }
            printBandwidth("d2d_c2c custom", data.dataSizeInBytes, timer.tocMs());
        }
    }

    return 0;
}
