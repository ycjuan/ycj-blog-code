#include <vector>
#include <random>
#include <cassert>
#include <algorithm>
#include "timer.cuh"

using namespace std;

/*
This program performs a "masked" vB = mA x vX matrix-vector multiplication,
where mA is M x N, vX is N x 1, vB is M x 1

There is a P x 1 vector vR used to indicate which rows should be processed and output to vB.

Note that I assume M >> N, as it is a more common case in my field (machine learning)
(for example - M is number of documents, and N is embedding dimension)

I compare the following data structures to store mA:
  - vector<vector<float>> (mA , where mA.size()  = M, and mA[0].size()  = N)
  - vector<vector<float>> (mAT, where mAT.size() = N, and mAT[0].size() = M)
  - vector<float> with ROW major (mAR)
  - vector<float> with COL major (mAC)

*/

int   M               = 10000;
int   N               = 100;
float DENSITY         = 1.0;
int   P               = int(M * DENSITY);
int   NUM_TRIALS      = 12;
int   NUM_TRIALS_WARM = 2; // Sometimes the first call to cuda may involve some context initialization, 
                           // so the timing could be inaccurate. To avoid such thing from happening, 
                           // we will ignore the first few trials for timing purpose

vector<float> cpu_row_A(vector<vector<float>> &mA, vector<float> &vX, vector<int> &vR, float &timeMs) {
    vector<float> vB(M, 0);
    Timer timer;
    for (int i : vR)
        for (int j = 0; j < N; j++)
            vB[i] += mA[i][j] * vX[j]; 
    timer.toc();
    timeMs = timer.getms();
    return vB;
}

__global__ void gpu_AR_kernel(float *d_mAR, float *d_vX, float *d_vB, int *d_vR, int M, int N, int P) {
    
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < P) {
        int i = d_vR[r];
        float b = 0;
        for (int j = 0; j < N; j++)
            b += d_mAR[i * N + j] * d_vX[j];
        d_vB[i] = b;
    }
}

vector<float> gpu_AR(float *d_mAR, float *d_vX, int *d_vR, float &timeMs) {
    vector<float> vB(M, 0);
    float *d_vB;
    cudaMalloc(reinterpret_cast<void**>(&d_vB), M * sizeof(float));

    int BLOCK_SIZE = 1024;
    int GRID_SIZE  = (int)ceil((double)P/BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);    
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    gpu_AR_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_mAR, d_vX, d_vB, d_vR, M, N, P);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeMs, start, stop);
    cudaEventDestroy(start);    
    cudaEventDestroy(stop);

    cudaMemcpy(vB.data(), d_vB, M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_vB);

    return vB;
}

__global__ void gpu_AC_kernel(float *d_mAC, float *d_vX, float *d_vB, int *d_vR, int M, int N, int P) {
    
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < P) {
        int i = d_vR[r];
        float b = 0;
        for (int j = 0; j < N; j++)
            b += d_mAC[j * M + i] * d_vX[j];
        d_vB[i] = b;
    }
}

vector<float> gpu_AC(float *d_mAC, float *d_vX, int *d_vR, float &timeMs) {
    vector<float> vB(M, 0);
    float *d_vB;
    cudaMalloc(reinterpret_cast<void**>(&d_vB), M * sizeof(float));

    int BLOCK_SIZE = 1024;
    int GRID_SIZE  = (int)ceil((double)P/BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);    
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    gpu_AC_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_mAC, d_vX, d_vB, d_vR, M, N, P);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeMs, start, stop);
    cudaEventDestroy(start);    
    cudaEventDestroy(stop);

    cudaMemcpy(vB.data(), d_vB, M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_vB);

    return vB;
}

void compare(vector<float> &vecA, vector<float> &vecB) {
    assert(vecA.size() == vecB.size());
    for (int i = 0; i < vecA.size(); i++) {
        if ( abs(vecA[i] - vecB[i]) > 1e-5 ) {
            printf("result inconsistency found.\n");
            return;
        }
    }
}

int main(void) {

    printf("P = %d\n", P);

    // declare mA, vX, vR
    vector<vector<float>> mA(M, vector<float>(N));
    vector<float>         vX(N);
    vector<int>           vR(M);

    // generate random values
    auto rng = std::default_random_engine {};
    uniform_real_distribution<> dist(0, 1.0);
    for (int i = 0; i < M; i++) {
        vR[i] = i;
        for (int j = 0; j < N; j++)
            mA[i][j] = dist(rng);
    }
    shuffle(vR.begin(), vR.end(), rng);
    vR.resize(P);
    sort(vR.begin(), vR.end());
    //for (int i = 0; i < 100; i++)
    //    printf("vR[%d]: %d\n", i, vR[i]);
    for (int j = 0; j < N; j++)
        vX[j] = dist(rng);

    // transform mA into mAT / mAR / mAC
    vector<vector<float>> mAT(N, vector<float>(M));
    vector<float> mAR(M * N);
    vector<float> mAC(M * N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            mAT[j][i] = mA[i][j];
            mAR[i * N + j] = mA[i][j];
            mAC[j * M + i] = mA[i][j];
        }
    }

    // copy mAR, mAC, vX, vR into GPU memory
    float *d_mAR;
    float *d_mAC;
    float *d_vX;
    int   *d_vR;
    cudaMalloc(reinterpret_cast<void**>(&d_mAR), M * N * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_mAC), M * N * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_vX ),     N * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_vR ),     M * sizeof(int));
    cudaMemcpy(d_mAR, mAR.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mAC, mAC.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vX ,  vX.data(),     N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vR ,  vR.data(),     M * sizeof(int)  , cudaMemcpyHostToDevice);

    // cpu_row_A
    vector<float> vB_cpu_row_A;
    float timeTotal = 0;
    for (int i = 0; i < NUM_TRIALS; i++) {
        float timeMs = 0;
        vB_cpu_row_A = cpu_row_A(mA, vX, vR, timeMs);
        if (i >= NUM_TRIALS_WARM)
            timeTotal += timeMs;
    }
    printf("time for cpu_row_A = %.2f ms\n", timeTotal / (NUM_TRIALS - NUM_TRIALS_WARM));

    // gpu_AR
    vector<float> vB_gpu_AR;
    timeTotal = 0;
    for (int i = 0; i < NUM_TRIALS; i++) {
        float timeMs = 0;
        vB_gpu_AR = gpu_AR(d_mAR, d_vX, d_vR, timeMs);
        if (i >= NUM_TRIALS_WARM)
            timeTotal += timeMs;
    }
    printf("time for gpu_AR = %.2f ms\n", timeTotal / (NUM_TRIALS - NUM_TRIALS_WARM));
    compare(vB_cpu_row_A, vB_gpu_AR);

    // gpu_AC
    vector<float> vB_gpu_AC;
    timeTotal = 0;
    for (int i = 0; i < NUM_TRIALS; i++) {
        float timeMs = 0;
        vB_gpu_AC = gpu_AC(d_mAC, d_vX, d_vR, timeMs);
        if (i >= NUM_TRIALS_WARM)
            timeTotal += timeMs;
    }
    printf("time for gpu_AC = %.2f ms\n", timeTotal / (NUM_TRIALS - NUM_TRIALS_WARM));
    compare(vB_cpu_row_A, vB_gpu_AC);

    // Free cuda
    cudaFree(d_mAR);
    cudaFree(d_mAC);
    cudaFree(d_vX);
    cudaFree(d_vR);

    // check if there is cuda error
    cudaDeviceSynchronize();
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
        printf("ERROR: there is a cuda error: %s\n", cudaGetErrorString(cudaError));
}
