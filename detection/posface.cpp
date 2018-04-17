
#include "posface.h"
posface::posface(){}

posface::posface(const std::string model_file, const std::string trained_file)
{
    #ifdef CPU_ONLY
        //caffe::SetMode(caffe::CPU, -1);
        Caffe::set_mode(Caffe::CPU);
        std::cout << '\n' << "USE CPU" << '\n';   // by yzh
    #else
        caffe::SetMode(caffe::GPU, 0);
        std::cout << '\n' << "USE GPU" << '\n';   // by yzh
    #endif

//    for(int i = 0; i < model_file.size(); i++)
//    {
    std::shared_ptr<Net<float>> net;

    cv::Size input_geometry;
    int num_channel;

    net_1.reset(new Net<float>(model_file,TEST));
    net_1->CopyTrainedLayersFrom(trained_file);

    Blob<float>* input_layer = net_1->blob_by_name("data").get();
    num_channel = input_layer->channels();
    input_geometry = cv::Size(input_layer->width(), input_layer->height());


    /*
     *         net.reset(new Net<float>(model_file[i], TEST));
    net->CopyTrainedLayersFrom(trained_file[i]);

    Blob<float>* input_layer = net->input_blobs()[0];
    num_channel = input_layer->channels();
    input_geometry = cv::Size(input_layer->width(), input_layer->height());
    */


    //nets_.push_back(net);
    //input_geometry_.push_back(input_geometry);

    num_channels_ = num_channel;
    std::cout << "insight make success " << std::endl;
     std::cout << "num channel "<< num_channels_ << std::endl;
//    if(i == 0){

//    }
//    else if(num_channels_ != num_channel)
//        std::cout << "Error: The number channels of the nets are different!" << std::endl;
//   // }
}

posface::~posface(){}


//vector<float> posface::GetFeature(cv::Mat croppedImages) {
const float* posface::GetFeature(cv::Mat croppedImages) {
    //std::shared_ptr<Net<float>> net = nets_[0];


    //std::vector<string> output_blob_names = output_blob_names_[i];

    //Blob<float>* input_layer = net->blob_by_name("data").get();
    Blob<float> *input_layer = net_1->input_blobs()[0];

    //std::cout <<"getFeature -->" << input_layer << std::endl;
    input_layer->Reshape(1, num_channels_,
                         croppedImages.rows, croppedImages.cols);

    /* Forward dimension change to all layers. */
    net_1->Reshape();

    std::vector<cv::Mat> input_channels;
    //WrapInputLayer(croppedImages, &input_channels, 0);
    net_1->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* features = net_1->output_blobs()[0];
    //Blob<float>* confidence = net->blob_by_name(output_blob_names[1]).get();
    //int count = confidence->count() / 2;

    const float* features_n = features->cpu_data();
    //std::cout << "feature1 : " << features_n[0] << ":"<< std::endl;
    return features_n;



    //std::cout << "Error: The number channels of the nets are different!" << std::endl;
 }
//cv::Mat findNonReflectiveTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points, cv::Mat &Tinv ) {
//  assert(source_points.size() == target_points.size());
//  assert(source_points.size() >= 2);
//  cv::Mat U = cv::Mat::zeros(target_points.size() * 2, 1, CV_64F);
//  cv::Mat X = cv::Mat::zeros(source_points.size() * 2, 4, CV_64F);
//  for (int i = 0; i < target_points.size(); i++) {
//    U.at<double>(i * 2, 0) = source_points[i].x;
//    U.at<double>(i * 2 + 1, 0) = source_points[i].y;
//    X.at<double>(i * 2, 0) = target_points[i].x;
//    X.at<double>(i * 2, 1) = target_points[i].y;
//    X.at<double>(i * 2, 2) = 1;
//    X.at<double>(i * 2, 3) = 0;
//    X.at<double>(i * 2 + 1, 0) = target_points[i].y;
//    X.at<double>(i * 2 + 1, 1) = -target_points[i].x;
//    X.at<double>(i * 2 + 1, 2) = 0;
//    X.at<double>(i * 2 + 1, 3) = 1;
//  }
//  cv::Mat r = X.inv(cv::DECOMP_SVD)*U;
//  Tinv = (cv::Mat_<double>(3, 3) << r.at<double>(0), -r.at<double>(1), 0,
//                       r.at<double>(1), r.at<double>(0), 0,
//                       r.at<double>(2), r.at<double>(3), 1);
//  cv::Mat T = Tinv.inv(cv::DECOMP_SVD);
//  Tinv = Tinv(cv::Rect(0, 0, 2, 3)).t();
//  return T(cv::Rect(0,0,2,3)).t();
//}

//cv::Mat findSimilarityTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points, cv::Mat &Tinv) {
//  cv::Mat Tinv1, Tinv2;
//  cv::Mat trans1 = findNonReflectiveTransform(source_points, target_points, Tinv1);
//  std::vector<cv::Point2d> source_point_reflect;
//  for (auto sp : source_points) {
//    source_point_reflect.push_back(cv::Point2d(-sp.x, sp.y));
//  }
//  cv::Mat trans2 = findNonReflectiveTransform(source_point_reflect, target_points, Tinv2);
//  trans2.colRange(0,1) *= -1;
//  std::vector<cv::Point2d> trans_points1, trans_points2;
//  transform(source_points, trans_points1, trans1);
//  transform(source_points, trans_points2, trans2);
//  double norm1 = norm(cv::Mat(trans_points1), cv::Mat(target_points), cv::NORM_L2);
//  double norm2 = norm(cv::Mat(trans_points2), cv::Mat(target_points), cv::NORM_L2);
//  Tinv = norm1 < norm2 ? Tinv1 : Tinv2;
//  return norm1 < norm2 ? trans1 : trans2;
//}



void posface::WrapInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, int i)
{


    unsigned start ;
    unsigned end ;

    start = clock();

     std::cout << "Run  times,   "<< (end-start)/1000000.0  <<std::endl;

    //Blob<float> *input_layer = nets_[i]->input_blobs()[0];
    Blob<float> *input_layer = net_1 ->input_blobs()[0];
    int width = input_layer->width();
    int height = input_layer->height();
    int num = input_layer->num();
    float *input_data = input_layer->mutable_cpu_data();

    for (int j = 0; j < num; j++) {
        //std::vector<cv::Mat> *input_channels;
        for (int k = 0; k < input_layer->channels(); ++k) {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels->push_back(channel);
            input_data += width * height;
        }
        cv::Mat img = imgs[j];
        cv::split(img, *input_channels);
        input_channels->clear();
    }
    end = clock();
    std::cout << "Run WrapInputLayer222,   "<< (end-start)/1000000.0  <<std::endl;
    std::cout << "Run WrapInputLayer 2222,   "<< (end-start)  <<std::endl;
}
void posface::WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels, int i)
{
    unsigned start ;
    unsigned end ;

    start = clock();

    //Blob<float>* input_layer = nets_[i]->input_blobs()[0];
    Blob<float> *input_layer = net_1 ->input_blobs()[0];
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int j = 0; j < input_layer->channels(); ++j)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }

    //cv::Mat sample_normalized;
    //cv::subtract(img, mean_[i], img);
    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(img, *input_channels);
    end = clock();
    std::cout << "Run WrapInputLayer 333,   "<< (end-start)/1000000.0  <<std::endl;
    std::cout << "Run WrapInputLayer 3333,   "<< (end-start)  <<std::endl;

}


//    auto netoutput = kCaffeBinding->Forward({ croppedImages }, net);
//    vector<float> result;
//    for (int i = 0; i < netoutput["fc1"].size[1]; i++) {
//        float this_rect = netoutput["fc1"].data[i];
//        result.push_back(this_rect);
//    }
//    return result;




//std::unordered_map<std::string, DataBlob> CaffeBinding::Forward(int net_id) {
//  if (!(*predictors_[net_id]).get()) {
//    auto predictor =
//      std::make_unique<caffe::Net<float>>(prototxts[net_id], Phase::TEST);
//    predictor->ShareTrainedLayersWith(nets_[net_id]);
//    (*predictors_[net_id]).reset(predictor.release());
//  }
//  auto* predictor = (*predictors_[net_id]).get();
//  const std::vector<Blob<float>*>& nets_output = predictor->ForwardPrefilled();
//  std::unordered_map<std::string, DataBlob> result;
//  for (int n = 0; n < nets_output.size(); n++) {
//    DataBlob blob = { nets_output[n]->cpu_data(), nets_output[n]->shape(), predictor->blob_names()[predictor->output_blob_indices()[n]] };
//    result[blob.name] = blob;
//  }
//  return result;
//}

//std::unordered_map<std::string, DataBlob> CaffeBinding::Forward(std::vector<cv::Mat>&& input_image, int net_id) {
//  SetMemoryDataLayer("data", move(input_image), net_id);
//  return Forward(net_id);
//}



//void MTCNN::Predict(const cv::Mat& img, int i)
//{
//    std::shared_ptr<Net<float>> net = nets_[i];
//	std::vector<string> output_blob_names = output_blob_names_[i];

//    Blob<float>* input_layer = net->blob_by_name("data").get();
//    input_layer->Reshape(1, num_channels_,
//                         img.rows, img.cols);
//    /* Forward dimension change to all layers. */
//    net->Reshape();

//    std::vector<cv::Mat> input_channels;
//    WrapInputLayer(img, &input_channels, i);
//    net->Forward();

//    /* Copy the output layer to a std::vector */
//    Blob<float>* rect = net->blob_by_name(output_blob_names[0]).get();
//    Blob<float>* confidence = net->blob_by_name(output_blob_names[1]).get();
//    int count = confidence->count() / 2;

//    const float* rect_begin = rect->cpu_data();
//    const float* rect_end = rect_begin + rect->channels() * count;
//    regression_box_temp_ = std::vector<float>(rect_begin, rect_end);

//    const float* confidence_begin = confidence->cpu_data() + count;
//    const float* confidence_end = confidence_begin + count;

//    confidence_temp_ = std::vector<float>(confidence_begin, confidence_end);
//}
