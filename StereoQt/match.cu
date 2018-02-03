#include "match.h"
#include "nan.h"
#include <algorithm>
#include <limits>
#include <iostream>

const int Match::OCCLUDED = std::numeric_limits<int>::max();

Match::Match(cv::Mat left, cv::Mat right, bool color)
{
	originalHeightL = imGetYSize(left);
	int height = std::min(imGetYSize(left), imGetYSize(right));
	imSizeL = Coord(imGetXSize(left), height);
	imSizeR = Coord(imGetXSize(right), height);

	imColorLeft = left;
	imColorRight = right;

	imColorLeftMin = imColorLeftMax = 0;//?
	imColorRightMin = imColorRightMax = 0;	//?

	dispMin = dispMax = 0;
	d_left = cv::Mat(imSizeL.y, imSizeL.x, CV_32SC1);
	vars0 = cv::Mat(imSizeL.y, imSizeL.x, CV_32SC1);
	varsA = cv::Mat(imSizeL.y, imSizeL.x, CV_32SC1);

	if (d_left.cols == 0 || vars0.cols == 0 || varsA.cols == 0)
	{
		std::cerr << "Not enough memory!" << std::endl;
		exit(1);
	}

}

void Match::InitMatch(cv::Mat& left, cv::Mat& right)
{
	originalHeightL = imGetYSize(left);
	int height = std::min(imGetYSize(left), imGetYSize(right));
	imSizeL = Coord(imGetXSize(left), height);
	imSizeR = Coord(imGetXSize(right), height);

	imColorLeft = left;
	imColorRight = right;

	imColorLeftMin = imColorLeftMax = 0;//?
	imColorRightMin = imColorRightMax = 0;	//?

	dispMin = dispMax = 0;
	d_left = cv::Mat(imSizeL.y, imSizeL.x, CV_32SC1);
	vars0 = cv::Mat(imSizeL.y, imSizeL.x, CV_32SC1);
	varsA = cv::Mat(imSizeL.y, imSizeL.x, CV_32SC1);

	if (d_left.cols == 0 || vars0.cols == 0 || varsA.cols == 0)
	{
		std::cerr << "Not enough memory!" << std::endl;
		exit(1);
	}

}

Match::~Match()
{
	// imfree
	imColorLeftMin.release();
	imColorLeftMax.release();
	imColorRightMin.release();
	imColorRightMax.release();

	d_left.release();
	vars0.release();
	varsA.release();
}

void Match::SetMehod(int m)
{
	assert(m == 0 || m == 1 || m == 2);
	switch (m)
	{
	case 0:
		method = SAD;
		break;
	case 1:
		method = NCC;
		break;
	case 2:
		method = GRAPH;
		break;
	default:
		break;
	}
}

void Match::SaveXLeft(const char* filename)
{
	Coord outSize(imSizeL.x, originalHeightL);

	cv::Mat out = cv::Mat(outSize.y, outSize.x, CV_8UC1);

	for (int w = 0; w < imSizeL.x; w++) {
		for (int h = 0; h < imSizeL.y; h++)
		{
			int d = d_left.at<int>(h, w);
			out.at<uchar>(h, w) = ((d == OCCLUDED) ? 0 : static_cast<uchar>(std::abs(d)));
		}
	}

	cv::imwrite(filename, out);
	out.release();

}

cv::Mat Match::GetResultDisparity()
{
	Coord outSize(imSizeL.x, originalHeightL);
	//this->out = cv::Mat(outSize.y, outSize.x, CV_8UC1);

	if (this->method == GRAPH) {
		for (int w = 0; w < imSizeL.x; w++) {
			for (int h = 0; h < imSizeL.y; h++)
			{
				int d = d_left.at<int>(h, w);
				out.at<uchar>(h, w) = ((d == OCCLUDED) ? 0 : static_cast<uchar>(std::abs(d)));
			}
		}
	}
	else {

	}

	return out;
}

void Match::SaveScaledXLeft(const char*filename, bool flag)
{

}

/// 设置disparity的搜索范围，初始时disparity map d_left设为最大值
void Match::SetDispRange(int dMin, int dMax)
{
	dispMin = dMin;
	dispMax = dMax;

	if (!(dispMin <= dispMax)) {
		std::cerr << "Error: wrong disparity range!\n" << std::endl;
		exit(1);
	}

	for (int w = 0; w < imSizeL.x; w++)
		for (int h = 0; h < imSizeL.y; h++)
			d_left.at<int>(h, w) = OCCLUDED;
}

cv::Mat Match::PerformMatchAllMethods(QProgressBar* progressBar)
{
	switch (this->method)
	{
	case SAD:
		RunSAD();
		break;
	case NCC:
		RunNCC();
		break;
	case GRAPH:
		KZ2(progressBar);
		break;
	default:
		KZ2(progressBar);
		break;
	}

	progressBar->setValue(100);

	return GetResultDisparity();
}

void Match::RunLocalCUDA(bool useSAD)
{
	int height = imColorLeft.rows, width = imColorLeft.cols;
	// host memory
	cv::Mat left, right;
	cv::cvtColor(imColorLeft, left, cv::COLOR_BGR2GRAY);
	cv::cvtColor(imColorRight, right, cv::COLOR_BGR2GRAY);

	// device memory
	uchar *d_left, *d_right, *d_out;
	cudaMalloc((void**)&d_left, width*height);
	cudaMalloc((void**)&d_right, width*height);
	cudaMalloc((void**)&d_out, width*height);

	cudaMemcpy(d_left, left.data, width*height, cudaMemcpyHostToDevice);
	cudaMemcpy(d_right, right.data, width*height, cudaMemcpyHostToDevice);
	cudaMemset(d_out, 0, width*height * sizeof(uchar));

	// launch kernel
	dim3 block_size, grid_size;
	block_size = dim3(32, 32, 1);
	grid_size = dim3((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y, 1);

	if (useSAD)
		SADMatch << <grid_size, block_size >> > (d_left, d_right, d_out, 64, 5, width, height);
	else
		NCCMatch << <grid_size, block_size >> > (d_left, d_right, d_out, 64, 5, width, height);
	// copy result back
	this->out.release();
	this->out = cv::Mat(height, width, CV_8UC1);
	cudaMemcpy((this->out).data, d_out, width*height * sizeof(uchar), cudaMemcpyDeviceToHost);

	cudaFree(d_left);
	cudaFree(d_right);
	cudaFree(d_out);
	left.release();
	right.release();

}

void Match::RunSAD()
{
	RunLocalCUDA(true);
}

void Match::RunNCC()
{
	RunLocalCUDA(false);
}