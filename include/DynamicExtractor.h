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



namespace ORB_SLAM2 {
    class DynamicExtractor {

    public:
        // need model location to load the model
        // only supports mask-rcnn for now
        DynamicExtractor(const std::string &strModelPath, int maxUsage=1,
                         float confThreshold = 0.5, float maskThreshold = 0.3);

        // compute dynamic mask for given frame
        void extractMask(const cv::Mat &frame, cv::Mat &dynamic_mask);

    private:
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

        std::vector<std::string> classes; // classId --> className
        std::unordered_set<std::string> dynamicClasses; // name of dynamic classes
        cv::dnn::Net net; // mask-rcnn model
        cv::Mat prevMask;

    };

}
#endif //DYNAMIC_ORB_SLAM2_DYNAMICEXTRACTOR_H
