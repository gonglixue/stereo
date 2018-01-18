//
// Created by gonglixue on 18-1-17.
//
// ref
// http://sourishghosh.com/2016/stereo-calibration-cpp-opencv/

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// imageList是list.txt文件名，该文件存放一系列所有视角图片的文件名
static void StereoCalib(const char* imageList, int nx, int ny, bool useUncalibrated)
{
    bool displayCorners = false;
    bool showUndistorted = true;
    bool isVerticalStereo = false;
    const int maxScale = 1;
    const float squareSize = 1.0f;
    FILE* f = fopen(imageList, "rt");       // list.txt
    int i, j, lr;
    int N = nx * ny;
    std::vector<std::string> imageNames[2]; // a list of leftimage and a list of rightimage
    std::vector<cv::Point3f> boardModel;
    std::vector< std::vector<cv::Point3f> > objectPoints;
    std::vector< std::vector<cv::Point2f> > points[2];  //points[0]的每个元素是个vector，该vector是每张图的图像点的vector
    std::vector< cv::Point2f >  corners[2];
    bool found[2] = {false, false};
    cv::Size imageSize;

    // read in the list of circle grids
    if(!f){
        std::cout << "can not open file " << imageList << std::endl;
        return;
    }

    // set up objectpoints coordinates
    for(i=0; i<ny; i++)
        for(j=0; j<nx; j++)
            boardModel.push_back(
                    cv::Point3f((float)(i*squareSize), (float)(j*squareSize), 0.0f)
            );

    i = 0;
    for(;;)
    {
        char buf[1024];
        lr = i % 2; //left or right
        if(lr == 0)
            found[0] = found[1] = false;

        if(!fgets(buf, sizeof(buf)-3, f))
            break;
        size_t len = strlen(buf);
        while(len > 0 && isspace(buf[len-1]))
            buf[--len] = '\0';
        if(buf[0] == '#')
            continue;

        cv::Mat img = cv::imread(buf, 0);
        if(img.empty())
            break;
        imageSize = img.size();
        imageNames[lr].push_back(buf);

        i++;

        // if we did not find board on the left image,
        // it does not make sense to find it on the right
        if(lr == 1 && !found[0])
            continue;

        // find circle grids and centers there in:
        for(int s=1; s<=maxScale; s++)
        {
            cv::Mat timg = img;
            if(s > 1)
                resize(img, timg, cv::Size(), s, s, cv::INTER_CUBIC);
            found[lr] = cv::findCirclesGrid(
                    timg,
                    cv::Size(nx, ny),
                    corners[lr],  // image coordinates
                    cv::CALIB_CB_ASYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING
            );

            if(found[lr] || s==maxScale){
                cv::Mat mcorners(corners[lr]);
                mcorners *= (1./s);
            }
            if(found[lr])
                break;
        }

        if(displayCorners){
            std::cout << buf << std::endl;
            cv::Mat cimg;
            cv::cvtColor(img, cimg, cv::COLOR_GRAY2BGR);

            // draw chessboard corners works for circle grids too
            cv::drawChessboardCorners(
                    cimg, cv::Size(nx, ny), corners[lr], found[lr]
            );
            cv::imshow("corners", cimg);
            if((cv::waitKey(0)&255) == 27)
                exit(-1);
        }
        else
            std::cout << '.';

        if(lr == 1 && found[0] && found[1]){
            objectPoints.push_back(boardModel);
            points[0].push_back(corners[0]);
            points[1].push_back(corners[1]);
        }

    }
    fclose(f);

    /*
     * =====================begin to calebrate the stereo cameras============
     *
     * */
    cv::Mat M1 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat M2 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat D1, D2, R, T, E, F;
    std::cout << "\nRunning stereo calibration ...\n";
    cv::stereoCalibrate(
            objectPoints,
            points[0],
            points[1],
            M1, D1, M2, D2,
            imageSize, R, T, E, F,
            cv::TermCriteria(
                    cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 100, 1e-5
            ),
            cv::CALIB_FIX_ASPECT_RATIO | cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_SAME_FOCAL_LENGTH
    );
    std::cout << "Done\n\n";

    /*
     * =======================calibration quality check======================
     * */
    std::vector< cv::Point3f> lines[2];
    double avgErr = 0;
    int nframes = (int)objectPoints.size();

    for(i=0; i<nframes; i++){
        std::vector<cv::Point2f>& pt0 = points[0][i];
        std::vector<cv::Point2f>& pt1 = points[1][i];

        cv::undistortPoints(pt0, pt0, M1, D1, cv::Mat(), M1);
        cv::undistortPoints(pt1, pt1, M2, D2, cv::Mat(), M2);
        cv::computeCorrespondEpilines(pt0, 1, F, lines[0]);
        cv::computeCorrespondEpilines(pt1, 2, F, lines[1]);

        for(j=0; j<N; j++){
            double err = fabs(
                    pt0[j].x * lines[1][j].x + pt0[j].y*lines[1][j].y + lines[1][j].z
            ) + fabs(
                    pt1[j].x * lines[0][j].x + pt1[j].y*lines[0][j].y + lines[0][j].z
            );
            avgErr += err;
        }
    }

    std::cout << "avg err = " << avgErr/(nframes * N) << std::endl;

    /*
     * =============================rectification=======================================
     */
     if(showUndistorted){
         cv::Mat R1, R2, P1, P2, map11, map12, map21, map22;
         if(!useUncalibrated){
             cv::stereoRectify(
                     M1, D1, M2, D2,
                     imageSize,
                     R, T, R1, R2, P1, P2,
                     cv::noArray(), 0
             );
             isVerticalStereo = fabs(P2.at<double>(1,3)) > fabs(P2.at<double>(0, 3));
             // precompute maps for cvRemap()
             cv::initUndistortRectifyMap(
                     M1, D1, R1, P1, imageSize, CV_16SC2, map11, map12
             );
             cv::initUndistortRectifyMap(
                     M2, D2, R2, P2, imageSize, CV_16SC2, map21, map22
             );
         }
         else{
            // use intrinsic parameters of each camera, bug compute the rectification transformation directly from the fundamental matrix
             std::vector<cv::Point2f> allpoints[2];
             for(i=0; i<nframes; i++){
                 std::copy(
                         points[0][i].begin(),
                         points[0][i].end(),
                         std::back_inserter(allpoints[0])
                 );

                 std::copy(
                         points[1][i].begin(),
                         points[1][i].end(),
                         std::back_inserter(allpoints[1])
                 );

             }
             cv::Mat F = cv::findFundamentalMat(
                     allpoints[0], allpoints[1], cv::FM_8POINT
             ); // 建立像素坐标之间的变换关系
             cv::Mat H1, H2;
             cv::stereoRectifyUncalibrated(
                     allpoints[0], allpoints[1],
                     F,
                     imageSize,
                     H1, H2,
                     3
             );

             R1 = M1.inv() * H1 * M1;
             R2 = M2.inv() * H2 * M2;

             cv::initUndistortRectifyMap(
                     M1, D1, R1, P1,
                     imageSize,
                     CV_16SC2,
                     map21, map22
             );

         }
         // rectify the images and find disparity maps
         cv::Mat pair;
         if(!isVerticalStereo)
             pair.create(imageSize.height, imageSize.width*2, CV_8UC3);
         else
             pair.create(imageSize.height*2, imageSize.width, CV_8UC3);

         // setup for finding stereo correspondences
//         cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(
//                 -64, 128, 11, 100, 1000,
//                 32, 0, 15, 1000, 16,
//                 cv::StereoSGBM::MODE_HH
//         );
         cv::StereoSGBM stereo(
                 -64, 128, 11, 100, 1000,
                 32, 0, 15, 1000, 16,
                 false
         );




         for(i=0; i<nframes; i++)
         {
             cv::Mat img1 = cv::imread(imageNames[0][i].c_str(), 0);
             cv::Mat img2 = cv::imread(imageNames[1][i].c_str(), 0);
             cv::Mat img1r, img2r, disp, vdisp;

             if(img1.empty() || img2.empty())
                 continue;

             cv::remap(img1, img1r, map11, map12, cv::INTER_LINEAR);
             cv::remap(img2, img2r, map21, map22, cv::INTER_LINEAR);

             if(!isVerticalStereo || useUncalibrated){
                 //stereo.compute(img1r, img2r, disp);
                 stereo.operator()(img1r, img2r, disp);
                 cv::normalize(disp, vdisp, 0, 256, cv::NORM_MINMAX, CV_8U);
                 cv::imshow("disparity", vdisp);
             }
             if(!isVerticalStereo)
             {
                 cv::Mat part = pair.colRange(0, imageSize.width);
                 cv::cvtColor(img1r, part, cv::COLOR_GRAY2BGR);
                 part = pair.colRange(imageSize.width, imageSize.width*2);
                 cv::cvtColor(img2r, part, cv::COLOR_GRAY2BGR);

                 for(j=0; j<imageSize.height; j+=16)
                     cv::line(
                             pair,
                             cv::Point(0, j),
                             cv::Point(imageSize.width*2, j),
                             cv::Scalar(0, 255, 0)
                     );
             }else{
                 cv::Mat part = pair.rowRange(0, imageSize.height);
                 cv::cvtColor(img1r, part, cv::COLOR_GRAY2BGR);
                 part = pair.rowRange(imageSize.height, imageSize.height*2);
                 cv::cvtColor(img2r, part, cv::COLOR_GRAY2BGR);

                 for(j=0; j<imageSize.height; j+=16)
                     cv::line(
                             pair,
                             cv::Point(j, 0),
                             cv::Point(j, imageSize.height*2),
                             cv::Scalar(0, 255, 0)
                     );
             }

             cv::imshow("rectified", pair);
             if((cv::waitKey()&255) == 27)
                 break;
         }

     }
}

int main(int argc, char** argv)
{
    int board_w = 9, board_h = 6;
    const char* board_list = "ch12_list.txt";
    if(argc == 4){
        board_list = argv[1];
        board_w = atoi(argv[2]);
        board_h = atoi(argv[3]);
    }

    StereoCalib(board_list, board_w, board_h, true);
    return 0;
}