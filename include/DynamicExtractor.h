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




class DynamicExtractor {

public:
    // need model location to load the model
    // only supports mask-rcnn for now
    DynamicExtractor(std::string textGraph, std::string modelWeights, std::string classesFile, std::string dynamicClassFile,
            float confThreshold = 0.5, float maskThreshold = 0.3);

    // compute dynamic mask for given frame
    void extractMask(cv::Mat &frame, cv::Mat &dynamic_mask);
private:
    // return non-zero if the corresponding class is dynamic
    bool is_dynamic(int classId) {
        return dynamicClasses.count(classes[classId]);
    }
    float confThreshold; // Confidence threshold
    float maskThreshold; // Mask threshold

    std::vector<std::string> classes; // classId --> className
    std::unordered_set<std::string> dynamicClasses; // name of dynamic classes
    cv::dnn::Net net; // mask-rcnn model
};


#endif //DYNAMIC_ORB_SLAM2_DYNAMICEXTRACTOR_H
