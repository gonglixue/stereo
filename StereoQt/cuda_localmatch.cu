#include "cuda_localmatch.cuh"

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__device__ int SADLoss(uchar* patch1, uchar* patch2, int patch_size)
{
	int result = 0;
	for (int i = 0; i < patch_size; i++)
	{
		for (int j = 0; j < patch_size; j++)
		{
			int temp = patch1[i*patch_size + j] - patch2[i*patch_size + j];
			result += ((temp > 0) ? temp : -temp);// cuda math
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
	/* --------- test sth strange --------------*/
	//uchar* left_patch = new uchar[1];
	//uchar* right_patch = new uchar[1];
	//left_patch[0] = left[idy*width + idx];
	//right_patch[0] = right[idy*width + idx];
	//uchar loss = SADLoss(left_patch, right_patch, 1);
	//delete[] left_patch;
	//delete[] right_patch;
	//out[idy*width + idx] = loss;
	/*---------------end test sth strange ---------------*/


	int patch_width = win_size * 2 + 1;
	uchar* left_patch = new uchar[patch_width * patch_width];
	uchar* right_patch = new uchar[patch_width * patch_width];
	// copy the window from left to left_patch
	for (int u = -win_size; u <= win_size; u++)		//x
	{
		for (int v = -win_size; v <= win_size; v++)		//y
		{
			int x_in_left = idx + u;
			int y_in_left = idy + v;
			int x_in_patch = win_size + u;
			int y_in_patch = win_size + v;
			left_patch[y_in_patch*patch_width + x_in_patch]
				= left[y_in_left*width + x_in_left];
		}
	}

	float minimum_loss = 10000;
	int min_u = idx;
	int r_v = idy;
	for (int r_u = idx - search_range; r_u <= idx; r_u++)
	{
		if (r_u < win_size || r_u>width - win_size)
			continue;


		// copy the window from right to right_patch
		for (int u = -win_size; u <= win_size; u++) {
			for (int v = -win_size; v <= win_size; v++) {
				int x_in_right = r_u + u;
				int y_in_right = r_v + v;
				int x_in_patch = win_size + u, y_in_patch = win_size + v;
				right_patch[y_in_patch*patch_width + x_in_patch]
					= right[y_in_right*width + x_in_right];
			}
		}

		float loss = SADLoss(left_patch, right_patch, patch_width);
		if (loss < minimum_loss) {
			minimum_loss = loss;
			min_u = r_u;
		}
	}

	// assign out
	delete[] left_patch;
	delete[] right_patch;
	out[idy*width + idx] = uchar(idx - min_u);

}

__global__ void NCCMatch(uchar* left, uchar* right, uchar* out,
	int search_range, int win_size, int width, int height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < search_range || idx >= (width - win_size))
		return;
	if (idy < win_size || idy >= (height - win_size))
		return;

	int patch_width = win_size * 2 + 1;
	uchar* left_patch = new uchar[patch_width * patch_width];
	uchar* right_patch = new uchar[patch_width * patch_width];

	float left_avg = 0;
	// copy the window from left to left_patch
	for (int u = -win_size; u <= win_size; u++) {
		for (int v = -win_size; v <= win_size; v++) {
			int x_in_left = idx + u, y_in_left = idy + v;
			int x_in_patch = win_size + u, y_in_patch = win_size + v;

			left_patch[y_in_patch*patch_width + x_in_patch]
				= left[y_in_left*width + x_in_left];

			left_avg += left[y_in_left*width + x_in_left];
		}
	}
	left_avg = left_avg / (patch_width * patch_width);

	float max_energy = -10000;
	int min_u = idx, r_v = idy;
	// compare
	for (int r_u = idx - search_range; r_u <= idx; r_u++) {
		if (r_u < win_size || r_u>width - win_size)
			continue;
		// copy the window from right to right_patch
		for (int u = -win_size; u <= win_size; u++) {
			for (int v = -win_size; v <= win_size; v++) {
				int x_in_right = r_u + u, y_in_right = r_v + v;
				int x_in_patch = win_size + u, y_in_patch = win_size + v;
				right_patch[y_in_patch*patch_width + x_in_patch]
					= right[y_in_right*width + x_in_right];
			}
		}

		float energy = NCCEnergy(left_patch, right_patch, left_avg, patch_width);
		//float energy = 0;
		if (energy > max_energy) {
			min_u = r_u;
			max_energy = energy;
		}

	}

	delete[] left_patch;
	delete[] right_patch;
	out[idy*width + idx] = uchar(idx - min_u);
	//printf("idx - min_u: %d\n", (idx - min_u));
	//out[idy*width + idx] = left_avg;
}
__device__ float NCCEnergy(const uchar* patch1, uchar* patch2, float left_avg,
	int patch_size)
{
	// cannot change patch1

	// calculate right avg
	float right_avg = 0;
	for (int i = 0; i < patch_size; i++) {
		for (int j = 0; j < patch_size; j++) {
			right_avg = right_avg + patch2[i*patch_size + j];
		}
	}
	right_avg = right_avg / (patch_size * patch_size);

	float numerator = 0, temp3 = 0, temp4 = 0;
	// numerator = Sigma(temp1 .* temp2)
	// temp3 = Sigma(temp1 .* temp1)
	// temp4 = Sigma(temp2 .* temp2)

	for (int i = 0; i < patch_size; i++) // y
	{
		for (int j = 0; j < patch_size; j++) {
			float temp1 = (patch1[i*patch_size + j] - left_avg); // patch1 .- left_avg = temp1 = patch1
			float temp2 = (patch2[i*patch_size + j] - right_avg); // patch2 .- right_avg = temp2

			numerator = numerator + temp1 * temp2;
			temp3 = temp3 + temp1 * temp1;
			temp4 = temp4 + temp2 * temp2;

		}
	}

	float result = numerator / rsqrtf(temp3 * temp4);
	return result;
	/*
	return result;
	*/

}
