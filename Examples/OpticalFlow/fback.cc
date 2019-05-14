//
// Created by horacehxw on 5/12/19.
//
#include <DynamicExtractor.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

using namespace cv;

bool in_img(int x, int y, Mat &img) {
    return x >=0 && y>=0 && x<img.rows && y<img.cols;
}

void propagate_mask(Mat &mask, Mat &next_mask, Mat &flow) {
    Mat map = Mat(flow.size(), CV_32FC2);
    for (int y = 0; y<flow.rows; y++) {
        for (int x = 0; x<flow.cols; x++) {
            Point2f f = flow.at<Point2f>(y, x);
//            if (mask.at<uchar>(y,x) > 0)
//                map.at<Point2f>(y,x) = Point2f(x, y);
//            else
            map.at<Point2f>(y,x) = Point2f(x+f.x, y+f.y);
        }
    }
    remap(mask, next_mask, map, Mat(), INTER_NEAREST, BORDER_CONSTANT, Scalar(255,255,255));
    next_mask = mask & next_mask;
}

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
                           double, const Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}


int main() {
    ORB_SLAM2::DynamicExtractor ex(
            "ModelsCNN/"
    );

    Mat img1 = imread("Examples/OpticalFlow/img3.png", CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread("Examples/OpticalFlow/img4.png", CV_LOAD_IMAGE_COLOR);
    Mat mask1, mask2;
    ex.extractMask(img1, mask1);
    ex.extractMask(img2, mask2);
    imwrite("Examples/OpticalFlow/img3_mask.png", mask1);
    imwrite("Examples/OpticalFlow/img4_mask.png", mask2);

    //Mat mask = imread("Examples/OpticalFlow/img1_mask.png", CV_8UC1);

    cvtColor(img1, img1, CV_BGR2GRAY);
    cvtColor(img2, img2, CV_BGR2GRAY);

    Mat flow;
    std::cout << img1.type() <<", " << img2.type() << ", " << CV_8UC1 << std::endl;
    Mat next_mask;



    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    calcOpticalFlowFarneback(img1, img2, flow, 0.5, 3, 20, 3, 5, 1.2, 0);
    propagate_mask(mask1, next_mask, flow);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    std::cout << ttrack << std::endl;



    Mat cflow;
    calcOpticalFlowFarneback(img1, img2, flow, 0.5, 3, 20, 3, 5, 1.2, 0);
    cvtColor(img1, cflow, COLOR_GRAY2BGR);
    drawOptFlowMap(flow, cflow, 16, 1.5, Scalar(0, 255, 0));
    imwrite("Examples/OpticalFlow/flow2.png", cflow);



    imwrite("Examples/OpticalFlow/img4_mask_wrap.png", next_mask);

    std::cout << next_mask.type() << std::endl;

    return 0;
}

