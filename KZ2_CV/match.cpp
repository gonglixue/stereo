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
		std::cerr << "Not enough memroty!" << std::endl;
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

void Match::SaveXLeft(const char* filename)
{
	Coord outSize(imSizeL.x, originalHeightL);

	cv::Mat out = cv::Mat(outSize.y, outSize.x, CV_8UC1);

	for (int w = 0; w < imSizeL.x; w++) {
		for (int h = 0; h < imSizeL.y; h++)
		{
			int d = d_left.at<int>(h, w);
			out.at<uchar>(h, w) = ((d == OCCLUDED) ? 0 : static_cast<uchar>(d));
		}
	}

	cv::imwrite(filename, out);
	out.release();

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