#ifndef __METHODS_CUH__
#define __METHODS_CUH__

#include "common.cuh"

void h2h_r2r(Data &data);
void h2h_r2c(Data &data);
void h2h_c2r(Data &data);
void h2h_c2c(Data &data);

void h2d_r2r(Data &data);
void h2d_r2c(Data &data);
void h2d_c2r(Data &data);
void h2d_c2c(Data &data);

void d2h_r2r(Data &data);
void d2h_r2c(Data &data);
void d2h_c2r(Data &data);
void d2h_c2c(Data &data);

void d2d_r2r(Data &data);
void d2d_r2c(Data &data);
void d2d_c2r(Data &data);
void d2d_c2c(Data &data);

#endif