// Version 0.1 //

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <algorithm>

#include "ShapeDescriptor.h"


using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
    //! [load_image]
    CommandLineParser parser(argc, argv, "{@input | data/Original/Camera_MV-GE31GM#A9FE75EF-Snapshot-20190724151000-180013002000.png | input image}"
                                         "{@XML_path | data/data.xml | path to XML data to train}"
                                         "{@XML_final_path | data/fdata.xml | path to final data for use}");



    auto input_image = imread(parser.get<String>("@input"), IMREAD_COLOR);
    CV_Assert(!input_image.empty());

    cv::Mat gray, blur;
    cvtColor(input_image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur, Size(5, 5), 0, 0, BORDER_DEFAULT);

    Mat bw1;
    adaptiveThreshold(~blur, bw1, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, (2 * 55 + 1), 7); //

    //ximgproc::niBlackThreshold(~src, bw2, 255, THRESH_BINARY, (2 * Block_size + 1), k);

    // Inverse image
    bitwise_not(bw1, bw1);
    //imshow("bw", bw1);

    // Smooth
    Mat edges;
    adaptiveThreshold(bw1, edges, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -3);

    dilate(edges, edges, Mat::ones(2, 2, CV_8UC1));

    Mat smooth;
    bw1.copyTo(smooth);
    cv::blur(smooth, smooth, Size(2, 2));
    smooth.copyTo(bw1, edges);

    // Find Contours
    Mat cdst;
    Canny(bw1, cdst, 100, 300, 3);

    vector<vector<Point>> contours;
    findContours(cdst, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
    vector<vector<Point>> contours_poly(contours.size());

    // Approxymation
    for (size_t i = 0; i < contours.size(); i++)    
        approxPolyDP(contours[i], contours_poly[i], 2, true);


    // choice needed contours
    vector<vector<Point>> contours_dst;
    for (size_t i = 0; i < contours_poly.size(); i++)
    {
        if (contourArea(contours_poly[i]) > 800)
            contours_dst.push_back(contours_poly[i]);
    }

    // [Draw contours]
    Mat dst = Mat::zeros(input_image.size(), CV_8UC3);
    Scalar color_red = Scalar(0, 0, 255);
    for (size_t i = 0; i < contours_dst.size(); i++)
    {
        //if (arcLength(contours_poly[i], true) < 300 && arcLength(contours_poly[i], true) > 100)
        drawContours(dst, contours_dst, (int)i, color_red, 1, 8);
    }

    cv::imshow("View", input_image);
    cv::imshow("Dst", dst);
    cv::waitKey(0);

    // [Testing class ShapeDescriptor]
    string path = parser.get<String>("@XML_path");
    string f_path = parser.get<String>("@XML_final_path");
    ShapeDescriptor shapeDescriptor(true, f_path, path);

    int selected = 4;
    Mat test = Mat::zeros(input_image.size(), CV_8UC3);
    drawContours(test, contours_dst, selected, color_red, -1, 8);
    cv::imshow("test", test);
    cv::waitKey();

    std::cout << "\nContour # = " << shapeDescriptor.trainClassify(contours_dst[selected]) << "\n"; // if return -1 contours not found in DB
    //std::cout << "\nContour # = " << shapeDescriptor.classify(contours_dst[contour_num]) << "\n"; // if return -1 contours not found in DB
    //shapeDescriptor.addShape(contours_dst[contour_num], 1);

    // [\Testing class ShapeDescriptor]

    waitKey(0);
}

