#pragma once

#include "data.cuh"

void methodReference(Data data);
void methodBaseline(Data data);
void methodResQuant(Data data, bool copyResidualFromHost);