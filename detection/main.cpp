#include <iostream>
#include "MTCNN.h"
#include "opencv2/opencv.hpp"
#include <time.h>
#include "posface.h"

#include <ctime>

using namespace std;
using namespace cv;

cv::Mat findNonReflectiveTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points, cv::Mat &Tinv ) {
  assert(source_points.size() == target_points.size());
  assert(source_points.size() >= 2);
  cv::Mat U = cv::Mat::zeros(target_points.size() * 2, 1, CV_64F);
  cv::Mat X = cv::Mat::zeros(source_points.size() * 2, 4, CV_64F);
  for (int i = 0; i < target_points.size(); i++) {
    U.at<double>(i * 2, 0) = source_points[i].x;
    U.at<double>(i * 2 + 1, 0) = source_points[i].y;
    X.at<double>(i * 2, 0) = target_points[i].x;
    X.at<double>(i * 2, 1) = target_points[i].y;
    X.at<double>(i * 2, 2) = 1;
    X.at<double>(i * 2, 3) = 0;
    X.at<double>(i * 2 + 1, 0) = target_points[i].y;
    X.at<double>(i * 2 + 1, 1) = -target_points[i].x;
    X.at<double>(i * 2 + 1, 2) = 0;
    X.at<double>(i * 2 + 1, 3) = 1;
  }
  cv::Mat r = X.inv(cv::DECOMP_SVD)*U;
  Tinv = (cv::Mat_<double>(3, 3) << r.at<double>(0), -r.at<double>(1), 0,
                       r.at<double>(1), r.at<double>(0), 0,
                       r.at<double>(2), r.at<double>(3), 1);
  cv::Mat T = Tinv.inv(cv::DECOMP_SVD);
  Tinv = Tinv(cv::Rect(0, 0, 2, 3)).t();
  return T(cv::Rect(0,0,2,3)).t();
}

cv::Mat findSimilarityTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points, cv::Mat &Tinv ) {
  cv::Mat Tinv1, Tinv2;
  cv::Mat trans1 = findNonReflectiveTransform(source_points, target_points, Tinv1);
  std::vector<cv::Point2d> source_point_reflect;
  for (auto sp : source_points) {
    source_point_reflect.push_back(cv::Point2d(-sp.x, sp.y));
  }
  cv::Mat trans2 = findNonReflectiveTransform(source_point_reflect, target_points, Tinv2);
  trans2.colRange(0,1) *= -1;
  std::vector<cv::Point2d> trans_points1, trans_points2;
  transform(source_points, trans_points1, trans1);
  transform(source_points, trans_points2, trans2);
  double norm1 = norm(cv::Mat(trans_points1), cv::Mat(target_points), cv::NORM_L2);
  double norm2 = norm(cv::Mat(trans_points2), cv::Mat(target_points), cv::NORM_L2);
  Tinv = norm1 < norm2 ? Tinv1 : Tinv2;
  return norm1 < norm2 ? trans1 : trans2;
}


int main() {

    vector<string> model_file = {
            "../model/det1.prototxt",
            "../model/det2.prototxt",
            "../model/det3.prototxt"
//            "../model/det4.prototxt"
    };

    vector<string> trained_file = {
            "../model/det1.caffemodel",
            "../model/det2.caffemodel",
            "../model/det3.caffemodel"
//            "../model/det4.caffemodel"
    };

    MTCNN mtcnn(model_file, trained_file);

    vector<Rect> rectangles;
    string img_path = "../result/twice.jpg";
    Mat img = imread(img_path);


    vector<float> confidence;
    vector<vector<cv::Point>> points;

	int count = 1;
	unsigned start = time(NULL);
	for(int i=0; i < count; i++) {
        //mtcnn.detection_TEST(img, rectangles);
        mtcnn.detection(img, rectangles,confidence,points);
	}
	unsigned end = time(NULL);
	unsigned ave = (end-start)*1000.0/count;
	std::cout << "Run " << count << " times, "<<"Average time:" << ave << std::endl; 

//    std::cout << "point size : " << points.size()  << std::endl;
//    for(int i = 0; i < rectangles.size(); i++)
//        rectangle(img, rectangles[i], Scalar(255, 0, 0));

//    for(int i=0 ; i < points.size();i++)
//    {
//        //cv::Point _p = points[i]
//        std::vector<cv::Point> temp_alignment = points[i];
//        std::cout <<" i count " << i << "--->"<< temp_alignment.size() << temp_alignment << std::endl;

//        for(int j =0 ; j < temp_alignment.size(); j++){
//            std::cout <<" j count " << j  << "----------"<< temp_alignment[j] <<std::endl;
//            cv::circle(img, temp_alignment[j],1, cv::Scalar(255,255,0),2);
//        }

//    }
//    for(auto &i : points){
//        for(auto &j : i){
//            std::cout <<" j count " << j  << "----------" <<std::endl;
//            cv::circle(img, j,1, cv::Scalar(255,255,0),2);
//        }
//    }

    cv::Mat crop_image_temp = img(rectangles[0]);

//    vector<string> insight_model_file = {
//            "../R34/model.prototxt"
//    };

//    vector<string> insight_trained_file = {
//            "../R34/model.caffemodel",
//    };
    //model.prototxt  model-r50-am.caffemodel

    std::string insight_model_file = "../R34/model.prototxt";


    std::string insight_trained_file = "../R34/model-r50-am.caffemodel";
    posface posface1(insight_model_file, insight_trained_file);

    vector<Point2d> target_points = { { 30.2946,51.6963 },{ 65.5318,51.5014 },{ 48.0252,71.7366 },{ 33.5493,92.3655 },{ 62.7299,92.2041 } };
    //vector<Point2d> source_points =  points[0].

    vector<Point2d> source_points (points[0].begin(), points[0].end());
    cv::Mat trans_inv;
    float standardx = rectangles[0].x;
    float standardy = rectangles[0].y;
    vector<Point2d> source_points2;
//    std::vector<cv::Point> temp_alignment;
//    for(auto &j : i)
//    {
//        temp_alignment.push_back(cv::Point(j.y, j.x));
//    }
//    alignment.push_back(std::move(temp_alignment));

    for (const auto& fp: source_points){
        source_points2.push_back(cv::Point2d(fp.x -standardx, fp.y -standardy));
    }
     //std::cout<<"x" << fp.x -standardx <<"y" << fp.y -standardy << ' ';
    std::cout<< source_points2 << std::endl;
    std::cout<< source_points << std::endl;
    std::cout<< target_points << std::endl;
     std::cout<< rectangles[0]<< std::endl;
      std::cout<< rectangles[0].x << rectangles[0].y << std::endl;
         std::cout<< points[0].max_size() << std::endl;


    Mat trans = findSimilarityTransform(source_points, target_points, trans_inv);
    //Mat trans = posface1.findSimilarityTransform(source_points, target_points);

    std::cout<< trans << std::endl;
    Mat cropImage;
    warpAffine(img, cropImage, trans, Size(112, 112));

    //imwrite("twice122.jpg", cropImage);
    //imshow("face2", cropImage);
    //waitKey(0);

    Mat img2 = imread("twice122.jpg");
    //imwrite("twice122.jpg", cropImage);

//    Point2f src_center(crop_image_temp.cols/2.0F, crop_image_temp.rows/2.0F);
//    Mat rot_matrix = getRotationMatrix2D(src_center, 90.0, 1.0);

//    float dummy_query_data[10] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
//    cv::Mat trans2 = cv::Mat(2, 3, CV_32F, dummy_query_data);
//    //Mat trans2 = {{1.0,1.0,1.0},{1.0,1.0,1.0}};
//    //Mat rotated_img(Size(crop_image_temp.size().height, crop_image_temp.size().width), crop_image_temp.type());
//    Mat rotated_img;
//    std::cout<< trans2 << std::endl;
//    std::cout<< rot_matrix << std::endl;
//    warpAffine(crop_image_temp, rotated_img, trans2, crop_image_temp.size());
    //posface1.GetFeature(cropImage);

    //waitKey(0);
    //imshow("cropImage", cropImage);
    //croppedImages.push_back(cropImage);
    //std::cout<< cropImage << std::endl;
//    imshow("face", crop_image_temp);
//    waitKey(0);


//    imshow("face2", cropImage);
//    waitKey(0);

//    unsigned start_ffff = time(NULL);

//    //mtcnn.detection(img, rectangles,confidence,points);

    //unsigned start_ffff = time(NULL);

    start = clock();
    const float* feature = posface1.GetFeature(img2);
    //unsigned end_ffff = time(NULL);

    end = clock();
    //unsigned ave_ffff = (end_ffff-start_ffff)*1000.0;
    const float* f_begin = feature ;
    const float* f_end = f_begin + 512;
    std::vector<float> feature_v = std::vector<float>(f_begin, f_end);


       std::cout <<"feature pointer"  << feature <<std::endl;
        std::cout <<"feature pointer float"  << f_begin <<std::endl;
     std::cout << "Run  times,   "<< (end-start)/1000000.0  <<std::endl;



     start = clock();
     feature = posface1.GetFeature(img2);
     //unsigned end_ffff = time(NULL);

     end = clock();


      std::cout << "Run  times,   "<< (end-start)/1000000.0  <<std::endl;
      start = clock();
     feature = posface1.GetFeature(img2);
      //unsigned end_ffff = time(NULL);

      end = clock();


       std::cout << "Run  times,   "<< (end-start)/1000000.0  <<std::endl;

       start = clock();
       feature = posface1.GetFeature(img2);
       //unsigned end_ffff = time(NULL);

       end = clock();


        std::cout << "Run  times,   "<< (end-start)/1000000.0  <<std::endl;

        start = clock();
         feature = posface1.GetFeature(img2);
        //unsigned end_ffff = time(NULL);

        end = clock();


         std::cout << "Run  times,   "<< (end-start)/1000000.0 <<std::endl;



   // std::cout << "Run " << ave_ffff << " times,  python "<< (start-start_ffff) <<std::endl;


//    for (const auto& ff: feature_v)
//      std::cout << ff << ' ';
//    return 0;





}

/*
 * 
int main() {


    //the vector used to input the address of the net model
    vector<string> model_file = {
            "../model/det1.prototxt",
            "../model/det2.prototxt",
            "../model/det3.prototxt"
//            "../model/det4.prototxt"
    };

    //the vector used to input the address of the net parameters
    vector<string> trained_file = {
            "../model/det1.caffemodel",
            "../model/det2.caffemodel",
            "../model/det3.caffemodel"
//            "../model/det4.caffemodel"
    };

    MTCNN mtcnn(model_file, trained_file);

//    VideoCapture cap(0);
    VideoCapture cap("../../0294_02_004_angelina_jolie.avi");

    VideoWriter writer;
    writer.open("../result/0294_02_004_angelina_jolie.avi",CV_FOURCC('M', 'J', 'P', 'G'), 25, Size(1280,720), true);

    Mat img;
    int frame_count = 0;
    while(cap.read(img))
    {
        vector<Rect> rectangles;
        vector<float> confidences;
        std::vector<std::vector<cv::Point>> alignment;
        mtcnn.detection(img, rectangles, confidences, alignment);

        for(int i = 0; i < rectangles.size(); i++)
        {
            int green = confidences[i] * 255;
            int red = (1 - confidences[i]) * 255;
            rectangle(img, rectangles[i], cv::Scalar(0, green, red), 3);
            for(int j = 0; j < alignment[i].size(); j++)
            {
                cv::circle(img, alignment[i][j], 5, cv::Scalar(255, 255, 0), 3);
            }
        }

        frame_count++;
        cv::putText(img, std::to_string(frame_count), cvPoint(3, 13),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, CV_AA);
        writer.write(img);
        imshow("Live", img);
        waitKey(1);
    }

    return 0;
}


*/
