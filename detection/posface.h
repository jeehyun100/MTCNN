//
// Created by Young on 2016/11/27.
//

//#define CPU_ONLY



#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace caffe;

class posface {
public:

    posface();
    posface(const std::string model_file, const std::string trained_file);
    ~posface();
    const float* GetFeature(cv::Mat croppedImages);
    void WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels, int i);
    void WrapInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, int i);

    //std::vector<std::shared_ptr<Net<float>>> nets_;

    std::shared_ptr<Net<float>> net_1;
    std::vector<cv::Size> input_geometry_;
    //cv::Mat findSimilarityTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points, cv::Mat &Tinv);
    //cv::Mat findNonReflectiveTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points, cv::Mat& Tinv );
    //cv::Mat findSimilarityTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points);
    //variable for the output of the neural network
    //std::vector<cv::Rect> regression_box_;
    std::vector<float> features;
    int num_channels_;



};
