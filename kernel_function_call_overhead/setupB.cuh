#ifndef SETUPB_CUH
#define SETUPB_CUH

#include "common.cuh"

class FuncRunnerB
{
public:
    __device__ void runFunc(Param param, size_t taskId)
    {
        func7(param, taskId);
    }

private:
    __device__ void func0(Param param, size_t taskId)
    {
        param.d_count[taskId]++;
    }

    __device__ void func1(Param param, size_t taskId)
    {
        if (taskId != 1)
            func0(param, taskId);
    }

    __device__ void func2(Param param, size_t taskId)
    {
        if (taskId != 2)
            func1(param, taskId);
    }

    __device__ void func3(Param param, size_t taskId)
    {
        if (taskId != 3)
            func2(param, taskId);
    }

    __device__ void func4(Param param, size_t taskId)
    {
        if (taskId != 4)
            func3(param, taskId);
    }

    __device__ void func5(Param param, size_t taskId)
    {
        if (taskId != 5)
            func4(param, taskId);
    }

    __device__ void func6(Param param, size_t taskId)
    {
        if (taskId != 6)
            func5(param, taskId);
    }

    __device__ void func7(Param param, size_t taskId)
    {
        if (taskId != 7)
            func6(param, taskId);
    }

    int dummy1 = 1;
    float dummy2 = 2.0f;
};

#endif // SETUPB_CUH