#ifndef COMMON_CUH
#define COMMON_CUH

enum Method
{
    METHOD0 = 0,
    METHOD1 = 1,
    METHOD2 = 2,
    METHOD3 = 3,
};

struct Param
{
    long *d_count;
    long dataSize;
    int numCountInc;
    int numTrials;
    Method method;
};

#endif // COMMON_CUH