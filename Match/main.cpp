#include "localmatch.h"

int main()
{
	cv::Mat left = cv::imread("./left.png", CV_8UC1);
	cv::Mat right = cv::imread("./right.png", CV_8UC1);

	//cv::Mat disparity = SADMatch(left, right, 64, 5);
	//cv::imshow("disp_sad", disparity);
	//cv::imwrite("disp_sad.png", disparity);

	cv::Mat disparity_ncc = NCCMatch(left, right, 64, 5);
	cv::imshow("disp_ncc", disparity_ncc);
	cv::imwrite("disp_ncc.png", disparity_ncc);
	cv::waitKey(0);
	return 0;
}