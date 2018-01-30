#include "localmatch.h"
#include <cv.h>
#include <cxcore.h>
#include <cvaux.h>

cv::Mat SADMatch(cv::Mat& left, cv::Mat& right, int search_range, int win_wsize, int win_hsize)
{
	// TODO assert size
	// ...

	cv::Mat disparity = cv::Mat::zeros(left.size(), CV_8UC1);

}

// left & right are gray image
// rectification后只要在x方向找
cv::Mat SADMatch(cv::Mat& left, cv::Mat& right, int search_range, int win_size)
{
	// TODO assert size
	// ...

	cv::Mat disparity = cv::Mat::zeros(left.size(), CV_8UC1);
	int width = left.cols;
	int height = left.rows;
	printf("begin sad match.\nsize:%d %d\n", height, width);

	for (int u = search_range; u < width - win_size; u++)  // x
	{
		for (int v = win_size; v < height - win_size; v++)
		{
			// (u,v)是patch中心
			// calculate the total values in a window centered at (u,v)
			float left_window_value = 0;
			//printf("u v: %d %d\n", u, v);

			cv::Mat left_patch = left(cv::Range(v - win_size, v + win_size + 1), cv::Range(u - win_size, u + win_size + 1));//第一个range是行，第二个range是列

			// search in the right to find the minium loss within certain range
			float minimum_loss = 10000;
			int min_u = u;
			for (int r_u = u - search_range; r_u <= u; r_u++)  // 要搜索右边的区域吗？leftx - rightx应该是一定大于0的吧
			{
				if (r_u < win_size|| r_u > width - win_size)
					continue;
				int r_v = v;

				cv::Mat right_patch = right(cv::Range(r_v - win_size, r_v + win_size + 1), cv::Range(r_u - win_size, r_u + win_size + 1));
				float loss = SADLoss(left_patch, right_patch);
				if (loss < minimum_loss)
				{
					minimum_loss = loss;
					min_u = r_u;
				}
					
			}

			disparity.at<uchar>(v, u) = uchar(u - min_u);
			//printf("disp: %d\n", u - min_u);

		}
	}

	return disparity;
}

float SADLoss(cv::Mat& patch1, cv::Mat& patch2)
{
	cv::Mat temp = cv::abs(patch1 - patch2);
	float result = cv::sum(temp)[0];
	return result;
	
}

cv::Mat NCCMatch(cv::Mat& left, cv::Mat& right, int search_range, int win_size)
{
	cv::Mat disparity = cv::Mat::zeros(left.size(), CV_8UC1);
	int width = left.cols;
	int height = left.rows;
	printf("begin ncc match.\nsize:%d %d\n", height, width);

	for (int u = search_range; u < width - win_size; u++)  // x
	{
		for (int v = win_size; v < height - win_size; v++)
		{
			// (u,v)是patch中心
			// calculate the total values in a window centered at (u,v)
			float left_window_value = 0;
			//printf("u v: %d %d\n", u, v);

			cv::Mat left_patch = left(cv::Range(v - win_size, v + win_size + 1), cv::Range(u - win_size, u + win_size + 1));//第一个range是行，第二个range是列

			// 左图以(u,v)为中心的patch的均值
			float left_avg = cv::sum(left_patch)[0] / powf(2 * win_size + 1, 2);
			
																															// search in the right to find the minium loss within certain range
			float max_energy = -10000;
			int min_u = u;
			for (int r_u = u - search_range; r_u <= u; r_u++)  // 要搜索右边的区域吗？leftx - rightx应该是一定大于0的吧
			{
				if (r_u < win_size || r_u > width - win_size)
					continue;
				int r_v = v;

				cv::Mat right_patch = right(cv::Range(r_v - win_size, r_v + win_size + 1), cv::Range(r_u - win_size, r_u + win_size + 1));
				float right_avg = cv::sum(right_patch)[0] / powf(2 * win_size + 1, 2);

				float energy = NCCEnergy(left_patch, right_patch, left_avg);
				if (energy > max_energy)
				{
					min_u = r_u;
					max_energy = energy;
				}

			}

			disparity.at<uchar>(v, u) = uchar(u - min_u);
			printf("set v,u(%d, %d)\n", v, u);

		}
	}

	return disparity;
}

float NCCEnergy(cv::Mat& patch1, cv::Mat& patch2, float left_avg)
{
	cv::Mat temp1, temp2;
	cv::subtract(patch1, cv::Scalar(left_avg), temp1);

	float right_avg = cv::sum(patch2)[0] / (patch2.cols * patch2.rows);
	cv::subtract(patch2, cv::Scalar(right_avg), temp2);

	float numerator = cv::sum(temp1.mul(temp2))[0];
	float temp3, temp4;
	temp3 = cv::sum(temp1.mul(temp1))[0];
	temp4 = cv::sum(temp2.mul(temp2))[0];

	float result = numerator / sqrt(temp3 * temp4);
	return result;
}


void test()
{
	
}