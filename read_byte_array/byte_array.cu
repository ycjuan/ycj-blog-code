#include <vector>
#include <random>
#include <stdexcept>
#include <cassert>

/*

This program performs the following operations:

- There are NUM_DOCS docs
- Each doc has NUM_BYTES bytes
- For each doc, we check how many bytes is >= 24

I use different data types including char, int16_t, int32_t, int64_t, int4, long4
to read the data, and compare there performance difference.

*/

using namespace std;

#define CHECK_CUDA(func)                            \
{                                                   \
    cudaError_t status = (func);                    \
    if (status != cudaSuccess) {                    \
        string error = "CUDA API failed at line "   \
            + to_string(__LINE__) + " with error: " \
            + cudaGetErrorString(status) + "\n";    \
        throw runtime_error(error);                 \
    }                                               \
}

const int BLOCK_SIZE = 1024;
const int NUM_DOCS   = 1000000;
const int NUM_BYTES  = 256;
const int NUM_TRIALS = 10;

__global__ void kernel_1_byte(char *d_src, char *d_dst, int numRows, int numBytes) {

    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if (i < numRows) {
        char count = 0;
        for (int j = 0; j < numBytes; j++)
            if (d_src[i * numBytes + j] >= 24)
                count++; 
        d_dst[i] = count;
    }
}

__global__ void kernel_2_byte(char *d_src, char *d_dst, int numRows, int numBytes) {

    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int dataTypeBytes = 2;
    int numDataTypes  = numBytes / dataTypeBytes;

    if (i < numRows) {
        char count = 0;
        int16_t *d_src_cast = reinterpret_cast<int16_t*>(d_src);
        for (int j = 0; j < numDataTypes; j++) {
            int16_t val = d_src_cast[i * numDataTypes + j];
            for (int k = 0; k < dataTypeBytes; k++) {
                if (char(val) >= 24)
                    count++;
                val = val >> 8;
            }
        }
        d_dst[i] = count;
    }
}

__global__ void kernel_4_byte(char *d_src, char *d_dst, int numRows, int numBytes) {

    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int dataTypeBytes = 4;
    int numDataTypes  = numBytes / dataTypeBytes;

    if (i < numRows) {
        char count = 0;
        int32_t *d_src_cast = reinterpret_cast<int32_t*>(d_src);
        for (int j = 0; j < numDataTypes; j++) {
            int32_t val = d_src_cast[i * numDataTypes + j];
            for (int k = 0; k < dataTypeBytes; k++) {
                if (char(val) >= 24)
                    count++;
                val = val >> 8;
            }
        }
        d_dst[i] = count;
    }
}

__global__ void kernel_8_byte(char *d_src, char *d_dst, int numRows, int numBytes) {

    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int dataTypeBytes = 8;
    int numDataTypes  = numBytes / dataTypeBytes;

    if (i < numRows) {
        char count = 0;
        int64_t *d_src_cast = reinterpret_cast<int64_t*>(d_src);
        for (int j = 0; j < numDataTypes; j++) {
            int64_t val = d_src_cast[i * numDataTypes + j];
            for (int k = 0; k < dataTypeBytes; k++) {
                if (char(val) >= 24)
                    count++;
                val = val >> 8;
            }
        }
        d_dst[i] = count;
    }
}

__global__ void kernel_16_byte(char *d_src, char *d_dst, int numRows, int numBytes) {

    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int dataTypeBytes = 16;
    int numDataTypes  = numBytes / dataTypeBytes;

    if (i < numRows) {
        char count = 0;
        int4 *d_src_cast = reinterpret_cast<int4*>(d_src);
        for (int j = 0; j < numDataTypes; j++) {
            int4 val = d_src_cast[i * numDataTypes + j];
            for (int k = 0; k < dataTypeBytes / 4; k++) {
                if (char(val.x) >= 24)
                    count++;
                val.x = val.x >> 8;
                if (char(val.y) >= 24)
                    count++;
                val.y = val.y >> 8;
                if (char(val.z) >= 24)
                    count++;
                val.z = val.z >> 8;
                if (char(val.w) >= 24)
                    count++;
                val.w = val.w >> 8;
            }
        }
        d_dst[i] = count;
    }
}

__global__ void kernel_32_byte(char *d_src, char *d_dst, int numRows, int numBytes) {

    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int dataTypeBytes = 32;
    int numDataTypes  = numBytes / dataTypeBytes;

    if (i < numRows) {
        char count = 0;
        long4 *d_src_cast = reinterpret_cast<long4*>(d_src);
        for (int j = 0; j < numDataTypes; j++) {
            long4 val = d_src_cast[i * numDataTypes + j];
            for (int k = 0; k < dataTypeBytes / 4; k++) {
                if (char(val.x) >= 24)
                    count++;
                val.x = val.x >> 8;
                if (char(val.y) >= 24)
                    count++;
                val.y = val.y >> 8;
                if (char(val.z) >= 24)
                    count++;
                val.z = val.z >> 8;
                if (char(val.w) >= 24)
                    count++;
                val.w = val.w >> 8;
            }
        }
        d_dst[i] = count;
    }
}

vector<char> method_gpu(char *d_src, char *d_dst, int dataTypeBytes) {
    
    int GRID_SIZE = (int)ceil((double)NUM_DOCS/BLOCK_SIZE);
    float timeMs;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);    
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int t = 0; t < NUM_TRIALS; t++) {
        if (dataTypeBytes == 1)
            kernel_1_byte<<<GRID_SIZE, BLOCK_SIZE>>>(d_src, d_dst, NUM_DOCS, NUM_BYTES);
        else if (dataTypeBytes == 2)
            kernel_2_byte<<<GRID_SIZE, BLOCK_SIZE>>>(d_src, d_dst, NUM_DOCS, NUM_BYTES);
        else if (dataTypeBytes == 4)
            kernel_4_byte<<<GRID_SIZE, BLOCK_SIZE>>>(d_src, d_dst, NUM_DOCS, NUM_BYTES);
        else if (dataTypeBytes == 8)
            kernel_8_byte<<<GRID_SIZE, BLOCK_SIZE>>>(d_src, d_dst, NUM_DOCS, NUM_BYTES);
        else if (dataTypeBytes == 16)
            kernel_16_byte<<<GRID_SIZE, BLOCK_SIZE>>>(d_src, d_dst, NUM_DOCS, NUM_BYTES);
        else if (dataTypeBytes == 32)
            kernel_32_byte<<<GRID_SIZE, BLOCK_SIZE>>>(d_src, d_dst, NUM_DOCS, NUM_BYTES);
        else
            assert(false);
    }
    CHECK_CUDA( cudaDeviceSynchronize() )
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeMs, start, stop);
    cudaEventDestroy(start);    
    cudaEventDestroy(stop);
   
    printf("dataTypeBytes = %d, time = %f ms\n", dataTypeBytes, timeMs);

    vector<char> vecDst(NUM_DOCS);
    CHECK_CUDA( cudaMemcpy(vecDst.data(), d_dst, vecDst.size() * sizeof(char), cudaMemcpyDeviceToHost) )
    return vecDst;
}

vector<char> method_cpu(vector<char> &vecSrc) {
    vector<char> vecDst(NUM_DOCS, 0);
    for (int i = 0; i < NUM_DOCS; i++) {
        char count = 0;
        for (int j = 0; j < NUM_BYTES; j++) {
            if (vecSrc[i * NUM_BYTES + j] >= 24)
                count++;
        }
        vecDst[i] = count;
    }
    return vecDst;
}

void compare(vector<char> &vecA, vector<char> &vecB) {
    assert(vecA.size() == vecB.size());
    for (int i = 0; i < vecA.size(); i++) {
        //printf("%d, %d\n", vecA[i], vecB[i]);
        assert(vecA[i] == vecB[i]);
    }
}

int main(void) {

    // Prepare input dataset
    vector<char> vecSrc(NUM_DOCS * NUM_BYTES);

    auto rng = std::default_random_engine {};
    uniform_int_distribution<> dist(0, 255);
    for (int i = 0; i < NUM_DOCS * NUM_BYTES; i++)
        vecSrc[i] = dist(rng);

    // GPU malloc and copy
    const char *h_src = vecSrc.data();
    char *d_src;
    char *d_dst;
    CHECK_CUDA( cudaMalloc((void**) &d_src, vecSrc.size() * sizeof(char)) )
    CHECK_CUDA( cudaMalloc((void**) &d_dst, NUM_DOCS * sizeof(char)) )
    CHECK_CUDA( cudaMemcpy(d_src, vecSrc.data(), vecSrc.size() * sizeof(char), cudaMemcpyHostToDevice) )

    // CPU version
    vector<char> output_cpu = method_cpu(vecSrc);

    // GPU - 1 byte
    vector<char> output_1_byte = method_gpu(d_src, d_dst, 1);
    compare(output_cpu, output_1_byte);

    // GPU - 2 byte
    vector<char> output_2_byte = method_gpu(d_src, d_dst, 2);
    compare(output_cpu, output_2_byte);

    // GPU - 4 byte
    vector<char> output_4_byte = method_gpu(d_src, d_dst, 4);
    compare(output_cpu, output_4_byte);

    // GPU - 8 byte
    vector<char> output_8_byte = method_gpu(d_src, d_dst, 8);
    compare(output_cpu, output_8_byte);

    // GPU - 16 byte
    vector<char> output_16_byte = method_gpu(d_src, d_dst, 16);
    compare(output_cpu, output_16_byte);

    // GPU - 32 byte
    vector<char> output_32_byte = method_gpu(d_src, d_dst, 32);
    compare(output_cpu, output_32_byte);

}
