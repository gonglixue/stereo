#ifndef __LOCAL_MATCH_H__
#define __LOCAL_MATCH_H__

#include <opencv2\opencv.hpp>

cv::Mat SADMatch(cv::Mat& left, cv::Mat& right, int search_range, int win_wsize, int win_hsize);
cv::Mat SADMatch(cv::Mat& left, cv::Mat& right, int search_range, int win_size);
float SADLoss(cv::Mat& patch1, cv::Mat& patch2);

cv::Mat NCCMatch(cv::Mat& left, cv::Mat& right, int search_range, int win_size);
float NCCEnergy(cv::Mat& patch1, cv::Mat& patch2, float left_avg);

#endif // !__LOCAL_MATCH_H__

