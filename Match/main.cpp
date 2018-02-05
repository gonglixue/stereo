#if 0
#include "localmatch.h"

cv::Mat mynorml(cv::Mat& input)
{
	cv::imshow("input", input);
	cv::Mat output;
	cv::normalize(input, output, 0, 255, cv::NORM_MINMAX);
	//cv::imshow("output", output);

	return output;
}



int main()
{
	cv::Mat left = cv::imread("./view1.png", CV_8UC1);
	cv::Mat right = cv::imread("./view5.png", CV_8UC1);

	//cv::Mat disparity = SADMatch(left, right, 64, 5);
	//cv::imshow("disp_sad", disparity);
	//cv::imwrite("disp_sad.png", disparity);

	cv::Mat disparity_ncc = NCCMatch(left, right, 64, 6);
	cv::Mat norm_result;
	cv::normalize(disparity_ncc, norm_result, 0, 255, cv::NORM_MINMAX);
	//cv::imshow("disp_ncc", disparity_ncc);
	cv::imwrite("ncc_win13.png", norm_result);

	//cv::Mat disp = cv::imread("disp_sad.png", 0);
	//cv::Mat norm_disp = mynorml(disp);
	//cv::imshow("norm", norm_disp);
	//cv::waitKey(0);
	return 0;
}
#endif