#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
/*
comment reference
http://blog.csdn.net/dcrmg/article/details/52929669
 */

using namespace std;

void help(char* argv[]) {}

int main(int argc, char* argv[])
{
	int n_boards = 0;  // number of images
	float image_sf = 0.5f;
	float delay = 1.0f;
	int board_w = 0;  // number of inner corners
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
	cv::Size board_sz = cv::Size(board_w, board_h);  // 多少个棋盘内角点

	vector< vector<cv::Point2f> > image_points;
	vector< vector<cv::Point3f> > object_points;

	// 循环直至获取包含n_boards个棋盘角点的图像
	cv::Size image_size;

    int input_count = 0;
	while (input_count < n_boards)
	{
		cv::Mat image0, image;
        char input_filename[100];
        sprintf(input_filename, "../calib_example/Dataset/Image%d.jpg", (input_count + 1));
		image0 = cv::imread(input_filename);

		image_size = image0.size();
		cv::resize(image0, image, cv::Size(), image_sf, image_sf, cv::INTER_LINEAR);  //变为原来大小的1/2

		// Find the board
		vector<cv::Point2f> corners;
		bool found = cv::findChessboardCorners(image, board_sz, corners);  // corners得到的是图像坐标吗？
		cv::drawChessboardCorners(image, board_sz, corners, found);

		if (found) {
			image ^= cv::Scalar::all(255);	// ?

			cv::Mat mcorners(corners);		// ?
			mcorners *= (1. / image_sf);    // mcorners原图中的图像坐标
			image_points.push_back(corners);
			object_points.push_back(vector<cv::Point3f>());
			vector<cv::Point3f>& opts = object_points.back();
			opts.resize(board_n);
			for (int j = 0; j < board_n; j++)
			{
				opts[j] = cv::Point3f((float)(j / board_w), (float)(j%board_w), 0.0f);  // 世界坐标是：棋盘平面z=0, 棋盘左上角为原点？
			}
			cout << "Collected our " << (int)image_points.size() << " of " <<
				n_boards << " needed chessboard images\n" << endl;

		}
		cv::imshow("calibration", image);

        input_count++;

		if ((cv::waitKey(120) & 255) == 27)
			return -1;
	}
	// END COLLECTION WHILE LOOP

	// 开始标定
	cv::Mat intrinsic_matrix, distortion_coeffs;
	double err = cv::calibrateCamera(
		object_points,
		image_points,
		image_size,
		intrinsic_matrix,
		distortion_coeffs,
		cv::noArray(),  //rotation vector
		cv::noArray(),  //translation vector
		cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_PRINCIPAL_POINT
	);

	// save
	cout << "*** DONE\n Reprojection error is " << err << "\n Save Intrinsics.xml and Distortions.xml files\n";
	cv::FileStorage fs("intrinsics.xml", cv::FileStorage::WRITE);

	fs << "image_width" << image_size.width << "image_height" << image_size.height
		<< "camera_matrix" << intrinsic_matrix << "distortion_coefficients"
		<< distortion_coeffs;
	fs.release();

	// Loading
    fs.open("intrinsics.xml", cv::FileStorage::READ);
    cout << "\nimage_with:" << (int)fs["image_width"];
    cout << "\nimage_height:" << (int)fs["image_height"];

    cv::Mat intrinsic_matrix_loaded, distortion_coeffs_loaded;
    fs["camera_matrix"] >> intrinsic_matrix_loaded;
    fs["distortion_coefficients"] >> distortion_coeffs_loaded;
    cout << "\nintrinsic matrix:" << intrinsic_matrix_loaded;
    cout << "\ndistortion coefficients:" << distortion_coeffs_loaded << endl;

    // undistort map
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(
            intrinsic_matrix_loaded,
            distortion_coeffs_loaded,
            cv::Mat(),
            intrinsic_matrix_loaded,
            image_size,
            CV_16SC2,
            map1,
            map2
    );

    // show raw and undistorted image
    for(int i=1; i<=20; i++){
        cv::Mat image, image0;
        char filename[100];
        sprintf(filename, "../calib_example/Dataset/Image%d.jpg", i);
        image0 = cv::imread(filename);

        if(image0.empty())
            break;

        cv::remap(
                image0,
                image,
                map1,
                map2,
                cv::INTER_LINEAR,
                cv::BORDER_CONSTANT,
                cv::Scalar()
        );
        cv::imshow("undistored", image);
        char save_filename[20];
        sprintf(save_filename, "../calib_example/undistored_Image%d.tif", i);
        cv::imwrite(save_filename, image);

        if((cv::waitKey(30) & 255) == 27)
            break;
    }

    return 0;

}