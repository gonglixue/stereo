#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include <stdio.h>
#include <math_functions.hpp>

typedef unsigned char uchar;


__global__ void addKernel(int *c, const int *a, const int *b);

__global__ void SADMatch(uchar* left, uchar* right, uchar* out, int search_range, int win_size, int width, int height);
__device__ int SADLoss(uchar* patch1, uchar* patch2, int patch_size);

__global__ void NCCMatch(uchar* left, uchar* right, uchar* out, int search_range, int win_size, int width, int height);
__device__ float NCCEnergy(uchar* patch1, uchar* patch2, float left_avg, int patch_size);

