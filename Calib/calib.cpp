#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

void help(char* argv[]) {}

int main(int argc, char* argv[])
{
	int n_boards = 0;  // 根据输入获得
	float image_sf = 0.5f;
	float delay = 1.0f;
	int board_w = 0;
	int board_h = 0;

	if (argc < 4 || argc > 6) {
		cout << "\nError: Wrong number of input parameters";
		return -1;
	}

	board_w = atoi(argv[1]);
	board_h = atoi(argv[2]);
	n_boards = atoi(argv[3]);
	if (argc > 4)
		delay = atof(argv[4]);
	if (argc > 5)
		image_sf = atof(argv[5]);

	int board_n = board_w * board_h;
	cv::Size board_sz = cv::Size(board_w, board_h);

	cv::VideoCapture capture(0);
	if (!capture.isOpened()) {
		cout << "couldn't open the camera\n";
		help(argv);
		return -1;
	}

	vector<vector<cv::Point2f>> image_points;
	vector<vector<cv::Point3f>> object_points;

	// 循环直至获取包含n_boards个棋盘角点的图像
	double last_captured_timestamp = 0;
	cv::Size image_size;

	while (image_points.size() < (size_t)n_boards)
	{

	}
}