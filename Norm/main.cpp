#include <opencv2/opencv.hpp>
#include <stdio.h>

void norm_all(int num)
{
	for (int i = 1; i <= num; i++)
	{
		char in_file[20];
		sprintf(in_file, "%d.png", i);
		cv::Mat input = cv::imread(in_file, 0);
		cv::Mat output;
		cv::normalize(input, output, 0, 255, cv::NORM_MINMAX);

		char out_file[20];
		sprintf(out_file, "norm_%d.png", i);
		cv::imwrite(out_file, output);

		printf("%s done\n", out_file);
	}
}

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		printf("input number of images.\n");
		exit(1);
	}

	//int number = atoi(argv[1]);
	//norm_all(number);
	cv::Mat input = cv::imread(argv[1]);
	cv::normalize(input, input, 0, 255, cv::NORM_MINMAX);
	cv::imwrite(argv[2], input);
}