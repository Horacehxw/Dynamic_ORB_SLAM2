//
// Created by horacehxw on 4/10/19.
//

#include <DynamicExtractor.h>

using namespace cv;

int main() {
    DynamicExtractor ex(
            "../../ModelsCNN/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt",
            "../../ModelsCNN/frozen_inference_graph.pb",
            "../../ModelsCNN/mscoco_labels.names",
            "../../ModelsCNN/dynamic.txt"
            );

    Mat frame = imread("./human.png", CV_LOAD_IMAGE_COLOR);
    Mat mask;
    ex.extractMask(frame, mask);

    imwrite("dynamic_mask.png", mask);

    return 0;
}