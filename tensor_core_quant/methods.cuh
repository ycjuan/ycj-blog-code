
#include "common.cuh"


void quantGpuCuda(Data data, Setting setting);

void quantCpu(Data data, Setting setting);

void quantWmmaSimple(Data data, Setting setting);

void quantWmmaUnroll(Data data, Setting setting);

void quantWmmaUnrollV2(Data data, Setting setting);