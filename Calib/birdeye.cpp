//
// Created by gonglixue on 18-1-16.
//
#include <opencv2/opencv.hpp>
#include <iostream>

// board_w board_h intrinsics.xml checker_image
int main(int argc, char* argv[])
{
    int         board_w = atoi(argv[1]);
    int         board_h = atoi(argv[2]);
    int         board_n = board_w * board_h;

    cv::Size    board_sz(board_w, board_h);
    cv::FileStorage fs(argv[3], cv::FileStorage::READ);
    cv::Mat intrinsic, distortion;

    fs["camera_matrix"] >> intrinsic;
    fs["distortion_coefficients"] >> distortion;
    if(!fs.isOpened() || intrinsic.empty() || distortion.empty())
    {
        std::cout << "Error: couldn't load intrinsic parameters from "
                  << argv[3] << std::endl;
        return -1;
    }
    fs.release();

    cv::Mat gray_image, image, image0 = cv::imread(argv[4], 1);
    if(image0.empty())
    {
        std::cout << "Error: couldn't load image" << argv[4] << std::endl;
        return -1;
    }

    // undistort our image
    // src, dst, intrinsic, distortion, new_intrinsic
    cv::undistort(image0, image, intrinsic, distortion, intrinsic);
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    // get che checkboard on the plane
    std::vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(
            image,  // input undistorted image
            board_sz,
            corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS
    );
    if(!found){
        std::cout << "couldn't acquire checkboard on " << argv[4]
                  << ", only found " << corners.size() << " of " <<board_n
                  <<" corners\n";
        return -1;
    }

    // ??
    // get subpixel accuracy on those corners
    cv::cornerSubPix(
            gray_image,     // input undistorted image
            corners,
            cv::Size(11, 11),
            cv::Size(-1, -1),
            cv::TermCriteria(
                    cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
                    30, 0.1
            )
    );

    // get image and object points:
    // object points are at (r,c):
    // (0,0), (board_w-1,0),(0,board_h-1),(board_w-1, board_h-1)棋盘图的四个角点
    // That means corners are at: corners[r*board_w + c]
    cv::Point2f objPts[4], imgPts[4];
    objPts[0].x = 0;
    objPts[0].y = 0;
    objPts[1].x = board_w - 1;
    objPts[1].y = 0;
    objPts[2].x = 0;
    objPts[2].y = board_h - 1;
    objPts[3].x = board_w - 1;
    objPts[3].y = board_h - 1;
    imgPts[0] = corners[0];
    imgPts[1] = corners[board_w - 1];
    imgPts[2] = corners[(board_h - 1) * board_w];
    imgPts[3] = corners[(board_h - 1) * board_w + board_w - 1];

    // draw the points in order: BGRY
    cv::circle(image, imgPts[0], 9, cv::Scalar(255, 0, 0));
    cv::circle(image, imgPts[1], 9, cv::Scalar(0, 255, 0));
    cv::circle(image, imgPts[2], 9, cv::Scalar(0, 0, 255));
    cv::circle(image, imgPts[3], 9, cv::Scalar(0, 255, 255));

    // draw the found chessboard
    cv::drawChessboardCorners(image, board_sz, corners, found);
    cv::imshow("checkers", image);

    // find the homography
    cv::Mat H = cv::getPerspectiveTransform(objPts, imgPts);

    // left user adjust the z height of the view
    double Z = 25;      // 相当于相机外参的translation z
    cv::Mat birds_image;
    for(;;){
        H.at<double>(2, 2) = Z;

        cv::warpPerspective(
                image,
                birds_image,
                H,
                image.size(),
                cv::WARP_INVERSE_MAP | cv::INTER_LINEAR,
                cv::BORDER_CONSTANT,
                cv::Scalar::all(0)      //fill with black
        );
        cv::imshow("birds_eye", birds_image);
        int key = cv::waitKey() & 255;

        if(key == 'u')
            Z += 0.5;
        else if(key == 'd')
            Z -= 0.5;
        else if(key == 27)
            break;
    }

    // show rotation and translation vector
    std::vector<cv::Point2f> image_points;
    std::vector<cv::Point3f> object_points;
    for(int i=0; i<4; i++)
    {
        image_points.push_back(imgPts[i]);
        object_points.push_back(cv::Point3f(objPts[i].x, objPts[i].y, 0));
    }

    cv::Mat rvec, tvec, rmat;
    cv::solvePnP(
            object_points,      // 3d points in object coordinate
            image_points,       // 2d points in image coordinates
            intrinsic,
            cv::Mat(),
            rvec,
            tvec
    );// solve extrinsic parameter
    cv::Rodrigues(rvec, rmat);  // convert to 3x3 rotation matrix

    // print
    std::cout << "intrinsics matrinx: \n" << intrinsic << std::endl;
    std::cout << "rotation matrix:\n" << rmat << std::endl;
    std::cout << "translation vector: \n" << tvec << std::endl;
    std::cout << "homography matrix: \n" << H << std::endl;
    std::cout << "inverted homography matrix: \n" << H.inv() << std::endl;

    return 1;




}
