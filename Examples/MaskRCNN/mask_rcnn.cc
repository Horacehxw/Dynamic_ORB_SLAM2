//
// Created by horacehxw on 3/30/19.
//


#include <fstream>
#include <sstream>
#include <iostream>
#include <string.h>
#include <unordered_set>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float maskThreshold = 0.3; // Mask threshold

vector<string> classes;
vector<Scalar> colors;
unordered_set<string> dynamicClasses;

bool is_dynamic(int classId);

void load_data_info();

// Draw the predicted bounding box
void drawBox(Mat& frame, int classId, float conf, Rect box);

// Draw the mask into frame
void drawMask(Mat &frame, int classId, Rect box, Mat &mask);

// Postprocess the neural network's output for each frame
void postprocess(Mat &frame, const vector<Mat> &outs, Mat &dynamic_mask);

// draw orb features on two images
void draw_feature();

void draw_feature() {
    Mat img_1 = imread ( "human.png", CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( "human2.png", CV_LOAD_IMAGE_COLOR );

    //-- 初始化
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // Ptr<FeatureDetector> detector = FeatureDetector::create(detector_name);
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create(descriptor_name);
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );

    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    Mat outimg1;
    Mat greyimg1;
    cvtColor(img_1, greyimg1, CV_BGR2GRAY);
    drawKeypoints( greyimg1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imshow("ORB特征点",outimg1);
    imwrite("orb_features.png", outimg1);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> matches;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, matches );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    // 仅供娱乐的写法
    min_dist = min_element( matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance;} )->distance;
    max_dist = max_element( matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance;} )->distance;

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector< DMatch > good_matches;
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            good_matches.push_back ( matches[i] );
        }
    }

    //-- 第五步:绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    drawMatches ( img_1, keypoints_1, img_2, keypoints_2, matches, img_match );
    drawMatches ( img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch );
    imshow ( "所有匹配点对", img_match );
    imwrite("all_matches.png", img_match);
    imshow ( "优化后匹配点对", img_goodmatch );
    imwrite("good_matches.png", img_goodmatch);
    waitKey(0);
}

// For each frame, extract the bounding box and mask for each detected object
void postprocess(Mat &frame, const vector<Mat> &outs, Mat &dynamic_mask)
{
    Mat outDetections = outs[0];
    Mat outMasks = outs[1];

    cout << "out Detection size = " << outDetections.size << endl;
    cout << "out Masks size = " << outMasks.size << endl;
    // Output size of masks is NxCxHxW where
    // N - number of detected boxes
    // C - number of classes (excluding background)
    // HxW - segmentation shape
    const int numDetections = outDetections.size[2];
    const int numClasses = outMasks.size[1];

    outDetections = outDetections.reshape(1, outDetections.total() / 7);
    cout << "out Detection after resize = " << outDetections.size << endl;

    dynamic_mask = Mat::zeros(frame.size(), CV_8U);
    Mat mat_ones = Mat(frame.size(), CV_8U, Scalar(255));
    for (int i = 0; i < numDetections; ++i)
    {
        float score = outDetections.at<float>(i, 2);
        if (score > confThreshold)
        {
            // Extract class id
            int classId = static_cast<int>(outDetections.at<float>(i, 1));

            // Extract bounding box
            int left = static_cast<int>(frame.cols * outDetections.at<float>(i, 3));
            int top = static_cast<int>(frame.rows * outDetections.at<float>(i, 4));
            int right = static_cast<int>(frame.cols * outDetections.at<float>(i, 5));
            int bottom = static_cast<int>(frame.rows * outDetections.at<float>(i, 6));

            left = max(0, min(left, frame.cols - 1));
            top = max(0, min(top, frame.rows - 1));
            right = max(0, min(right, frame.cols - 1));
            bottom = max(0, min(bottom, frame.rows - 1));
            Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

            // Extract the mask for the object
            Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId));
            // Resize the mask, threshold, color and apply it on the image
            resize(objectMask, objectMask, Size(box.width, box.height));
            // threshold mask into binary 255/0 mask
            Mat mask = (objectMask > maskThreshold);
            mask.convertTo(mask, CV_8U);

            // Draw bounding box, colorize and show the mask on the image
            drawBox(frame, classId, score, box);

            // Draw mask
            drawMask(frame, classId, box, mask);

            if (is_dynamic(classId)) {
                mat_ones(box).copyTo(dynamic_mask(box), mask);
            }
        }
    }
}

// Draw the predicted bounding box, colorize and show the mask on the image
void drawBox(Mat& frame, int classId, float conf, Rect box)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    box.y = max(box.y, labelSize.height);
    rectangle(frame, Point(box.x, box.y - round(1.5*labelSize.height)), Point(box.x + round(1.5*labelSize.width), box.y + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

void drawMask(Mat &frame, int classId, Rect box, Mat &mask) {
    Scalar color = colors[classId%colors.size()];

    // make colorful mask
    Mat colorRoi = Mat(Size(box.width, box.height), frame.type(), color);
    Mat colorMask = Mat::zeros(Size(box.width, box.height), frame.type());
    colorRoi.copyTo(colorMask, mask);
    // color the segmentation and apply to image
    addWeighted(frame(box), 0.5, colorMask, 0.5, 0.0, frame(box));
}

bool is_dynamic(int classId) {
    return dynamicClasses.count(classes[classId]);
}

void load_data_info() {
    // Load names of classes
    string classesFile = "./mscoco_labels.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // load names of dynamic classes
    string dynamicClassFile = "./mscoco_labels.names";
    ifstream ifs2(dynamicClassFile.c_str());
    while (getline(ifs2, line)) dynamicClasses.insert(line);

//    // test if every dynamic class refer to existing class names
//    for (auto s: dynamicClasses) {
//        bool flag = false;
//        for (auto s_ : classes) {
//            if (s == s_) {
//                flag = true;
//            }
//        }
//        if (!flag) {
//            cout << s <<" mismatch" << endl;
//        }
//    }

    // Load the colors
    string colorsFile = "./colors.txt";
    ifstream colorFptr(colorsFile.c_str());
    while (getline(colorFptr, line))
    {
        char* pEnd;
        double r, g, b;
        r = strtod(line.c_str(), &pEnd);
        g = strtod(pEnd, NULL);
        b = strtod(pEnd, NULL);
        Scalar color = Scalar(r, g, b, 255.0);
        colors.push_back(Scalar(r, g, b, 255.0));
    }
}

int main(int argc, char *argv[])
{
    load_data_info();
    // Give the configuration and weight files for the model
    String textGraph = "./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
    String modelWeights = "./mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb";

    // Load the network
    Net net = readNetFromTensorflow(modelWeights, textGraph);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Open a video file or an image file or a camera stream.
    string str, outputFile;
    outputFile = "mask.png";
    VideoCapture cap;//根据摄像头端口id不同，修改下即可
    VideoWriter video;
    Mat frame, blob;
    //frame = imread("human.png", CV_LOAD_IMAGE_COLOR);
    cap.open("./human.png");


    // Create a window
    static const string kWinName = "Mask R-CNN in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);


    // get frame from the video
    cap >> frame;

    // Stop the program if reached end of video
    if (frame.empty())
    {
        cout << "Done processing !!!" << endl;
        cout << "Output file is stored as " << outputFile << endl;
        waitKey(3000);
        //break;
    }
    // Create a 4D blob from a frame.
    blobFromImage(frame, blob, 1.0, Size(frame.cols, frame.rows), Scalar(), true, false);
    //blobFromImage(frame, blob);
    cout << "blob size = " << blob.size << endl;
    //Sets the input to the network
    net.setInput(blob);

    // Runs the forward pass to get output from the output layers
    std::vector<String> outNames(2);
    outNames[0] = "detection_out_final";
    outNames[1] = "detection_masks";
    vector<Mat> outs;
    net.forward(outs, outNames);
    for (auto m: outs) {
        cout << "m.size = " << m.size << endl;
    }

    // Extract the bounding box and mask for each of the detected objects
    Mat dynamic_mask;
    postprocess(frame, outs, dynamic_mask);

    // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = format("Mask-RCNN on 3.6 GHz Intel Core i7 CPU, Inference time for a frame : %0.0f ms", t);
    putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

    // Write the frame with the detection boxes
    imwrite("dynamic_mask.png", dynamic_mask);
    imwrite(outputFile, frame);
    imshow(kWinName, frame);


    cap.release();
    return 0;
}

