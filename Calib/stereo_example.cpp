//
// Created by gonglixue on 18-1-17.
//

#include <opencv2/opencv.hpp>
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
                    cv::Size_(nx, ny),
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
}