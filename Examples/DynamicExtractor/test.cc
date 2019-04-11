//
// Created by horacehxw on 4/10/19.
//

#include <DynamicExtractor.h>
#include <chrono>

using namespace cv;

int main() {

    ORB_SLAM2::DynamicExtractor ex(
            "ModelsCNN/"
            );

    Mat frame = imread("Examples/DynamicExtractor/human.png", CV_LOAD_IMAGE_COLOR);
    Mat mask;

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    ex.extractMask(frame, mask);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    std::cout << ttrack << std::endl;

    imwrite("dynamic_mask.png", mask);

    return 0;
}