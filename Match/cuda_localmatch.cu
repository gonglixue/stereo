#include "cuda_localmatch.cuh"

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__device__ float SADLoss(uchar* patch1, uchar* patch2, int patch_size)
{
	float result = 0;
	for (int i = 0; i < patch_size; i++)
	{
		for (int j = 0; j < patch_size; j++)
		{
			float temp = patch1[i*patch_size + j] - patch2[i*patch_size + j];
			result += fabsf(temp);	// cuda math
		}
	}
	return result;
}

__global__ void SADMatch(uchar* left, uchar* right, uchar* out, 
	int search_range, int win_size, int width, int height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < search_range || idx >= (width - win_size))
		return;
	if (idy < win_size || idy >= (height - win_size))
		return;


}

__global__ void NCCMatch(uchar* left, uchar* right, uchar* out, int search_range, int win_size, int width, int height)
{

}
__device__ float NCCEnergy(uchar* patch1, uchar* patch2, float left_avg, int patch_size)
{

}
