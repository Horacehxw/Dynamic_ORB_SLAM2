//
// Created by horacehxw on 4/10/19.
//

#include <DynamicExtractor.h>
#include <chrono>

using namespace cv;
using namespace std;

void draw_feature(Mat& img_1, Mat& mask) {
    Mat mat_zeros = Mat::zeros(img_1.size(), CV_8UC3);
    Mat img_masked;
    Mat img_masked2;
    img_1.copyTo(img_masked, mask);
//    img_1.copyTo(img_masked2, mask);
//    imshow("1-mask", img_masked);
//    imshow("mask", img_masked2);
//    waitKey(0);

    //-- 初始化
    std::vector<KeyPoint> keypoints_1, keypoints_1_, keypoints_2, keypoints_3, keypoints_4;
    Ptr<FeatureDetector> detector = ORB::create();

    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1);
    detector->detect ( img_1,keypoints_1_);
    detector->detect ( img_masked,keypoints_2);
    detector->detect ( img_1, keypoints_3, mask);
    detector->detect ( img_1, keypoints_4, 1-mask);

    Mat outimg;
    Mat greyimg1;
    Mat greyimg_masked;
    cvtColor(img_1, greyimg1, CV_BGR2GRAY);
    cvtColor(img_masked, greyimg_masked, CV_BGR2GRAY);


    KeyPointsFilter filter;
    filter.runByPixelsMask(keypoints_1, mask);
    filter.runByPixelsMask(keypoints_1_, 1-mask);
    drawKeypoints( greyimg1, keypoints_1, outimg, Scalar(0,255,0), DrawMatchesFlags::DEFAULT);
    drawKeypoints( outimg, keypoints_1_, outimg, Scalar(0,0,255), DrawMatchesFlags::DEFAULT);

    imwrite("Examples/DynamicExtractor/outimg_1.png", outimg);

    drawKeypoints( greyimg_masked, keypoints_2, outimg, Scalar(0,255,0), DrawMatchesFlags::DEFAULT);
    imwrite("Examples/DynamicExtractor/outimg_2.png", outimg);

    drawKeypoints( greyimg1, keypoints_3, outimg, Scalar(0,255,0), DrawMatchesFlags::DEFAULT);
    imwrite("Examples/DynamicExtractor/outimg_3.png", outimg);

    drawKeypoints( greyimg1, keypoints_4, outimg, Scalar(0,0,255), DrawMatchesFlags::DEFAULT);
    drawKeypoints( outimg, keypoints_1, outimg, Scalar(0,255,0), DrawMatchesFlags::DEFAULT);
    imwrite("Examples/DynamicExtractor/outimg_false.png", outimg);
}

int main() {
    ORB_SLAM2::DynamicExtractor ex(
            "ModelsCNN/"
            );

    Mat frame = imread("Examples/DynamicExtractor/back.png", CV_LOAD_IMAGE_COLOR);
    Mat mask;

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    ex.extractMask(frame, mask);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    std::cout << ttrack << std::endl;

    imwrite("dynamic_mask.png", mask);

    //cvtColor(mask, mask,  CV_BGR2GRAY);
    draw_feature(frame, mask);

    return 0;
}