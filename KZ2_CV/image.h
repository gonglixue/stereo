#ifndef IMAGE_H
#define IMAGE_H

#include <stdlib.h>
#include <opencv2\opencv.hpp>

typedef enum
{
	IMAGE_GRAY,
	IMAGE_RGB,
	IMAGE_INT,
	IMAGE_FLOAT
}ImageType;

inline cv::Vec3b imgRef(cv::Mat im, int x, int y)
{
	cv::Vec3b value = im.at<cv::Vec3b>(x, y);
}

inline int imGetXSize(cv::Mat im)
{
	return im.cols;
}

inline int imGetYSize(cv::Mat im)
{
	return im.rows;
}




#endif
