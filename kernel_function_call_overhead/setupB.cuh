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
        param.d_count[taskId] = taskId;
        for (int i = 0; i < param.numCountInc; i++)
        {
            param.d_count[taskId] += 1;
        }
    }

    __device__ void func1(Param param, size_t taskId)
    {
        func0(param, taskId);
    }

    __device__ void func2(Param param, size_t taskId)
    {
        func1(param, taskId);
    }

    __device__ void func3(Param param, size_t taskId)
    {
        func2(param, taskId);
    }

    __device__ void func4(Param param, size_t taskId)
    {
        func3(param, taskId);
    }

    __device__ void func5(Param param, size_t taskId)
    {
        func4(param, taskId);
    }

    __device__ void func6(Param param, size_t taskId)
    {
        func5(param, taskId);
    }

    __device__ void func7(Param param, size_t taskId)
    {
        func6(param, taskId);
    }

    int dummy1 = 1;
    float dummy2 = 2.0f;
};

#endif // SETUPB_CUH