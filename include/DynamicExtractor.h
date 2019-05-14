//
// Created by horacehxw on 4/10/19.
//

#ifndef DYNAMIC_ORB_SLAM2_DYNAMICEXTRACTOR_H
#define DYNAMIC_ORB_SLAM2_DYNAMICEXTRACTOR_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <string.h>
#include <string>
#include <vector>
#include <unordered_set>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>



namespace ORB_SLAM2 {
    class DynamicExtractor {

    public:
        // need model location to load the model
        // only supports mask-rcnn for now
        DynamicExtractor(const std::string &strModelPath, int maxUsage=1, bool useOpticalFlow=false,
                         float confThreshold = 0.5, float maskThreshold = 0.3);

        // compute dynamic mask for given frame
        void extractMask(const cv::Mat &frame, cv::Mat &dynamic_mask);

    private:
        inline static void propagate_mask(cv::Mat &mask, cv::Mat &next_mask, cv::Mat &flow) {
            cv::Mat map = cv::Mat(flow.size(), CV_32FC2);
            for (int y = 0; y<flow.rows; y++) {
                for (int x = 0; x<flow.cols; x++) {
                    cv::Point2f f = flow.at<cv::Point2f>(y, x);
//            if (mask.at<uchar>(y,x) > 0)
//                map.at<Point2f>(y,x) = Point2f(x, y);
//            else
                    map.at<cv::Point2f>(y,x) = cv::Point2f(x+f.x, y+f.y);
                }
            }
            remap(mask, next_mask, map, cv::Mat(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(255));
            next_mask = mask & next_mask;
        }


        // do real mask extraction under CNN
        cv::Mat extractMask(const cv::Mat &frame);
        // return non-zero if the corresponding class is dynamic
        bool is_dynamic(int classId) {
            return dynamicClasses.count(classes[classId]);
        }

        float confThreshold; // Confidence threshold
        float maskThreshold; // Mask threshold
        int maskUsage; // prevMask usage counter
        int maxUsage; // max number of mask synchronization
        bool useOpticalFlow;
        std::vector<std::string> classes; // classId --> className
        std::unordered_set<std::string> dynamicClasses; // name of dynamic classes
        cv::dnn::Net net; // mask-rcnn model
        cv::Mat prevMask;
        cv::Mat prevFrame;

    };

}
#endif //DYNAMIC_ORB_SLAM2_DYNAMICEXTRACTOR_H
