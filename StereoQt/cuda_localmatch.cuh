#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define __CUDA_INTERNAL_COMPILATION__
#include "math_functions.h"
#undef __CUDA_INTERNAL_COMPILATION__

typedef unsigned char uchar;


__global__ void addKernel(int *c, const int *a, const int *b);

__global__ void SADMatch(uchar* left, uchar* right, uchar* out, int search_range, int win_size, int width, int height);
__device__ int SADLoss(uchar* patch1, uchar* patch2, int patch_size);

__global__ void NCCMatch(uchar* left, uchar* right, uchar* out, int search_range, int win_size, int width, int height);
__device__ float NCCEnergy(const uchar* patch1, uchar* patch2, float left_avg, int patch_size);

