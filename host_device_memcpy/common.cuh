#ifndef __COMMON_CUH__
#define __COMMON_CUH__

typedef float T;

struct Data
{
    T *h_data_src;
    T *h_data_dst;    
    T *hp_data_src;
    T *hp_data_dst;
    T *d_data_src;
    T *d_data_dst;    

    size_t numRows;
    size_t numCols;
    size_t dataSize;
    size_t dataSizeInBytes;

    void malloc(size_t numRows, size_t numCols);
    void print();
};

#endif