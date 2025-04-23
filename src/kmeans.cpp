#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <random>

using namespace cv;
using namespace std;


void segmKMeans(cv::Mat& image, const std::vector<cv::Point>& activationPoints) {
    cv::Mat samples = cv::Mat(image.rows * image.cols, 3, CV_32F);
    
    // Prepare data for k-means
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            for (int z = 0; z < 3; z++) {
                samples.at<float>(y + x * image.rows, z) = image.at<cv::Vec3b>(y, x)[z];
            }
        }
    }
    
    // Number of clusters
    int clusterCount = 10   ;
    cv::Mat labels;
    int attempts = 5;
    cv::Mat centers;
    
    // Using k-means
    cv::kmeans(samples, clusterCount, labels, 
              cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 10000, 0.0001), 
              attempts, cv::KMEANS_PP_CENTERS, centers);
    
    // Find which clusters to activate based on input points
    std::unordered_set<int> activeClusters;
    for (const cv::Point& pt : activationPoints) {
        if (pt.x >= 0 && pt.x < image.cols && pt.y >= 0 && pt.y < image.rows) {
            int idx = pt.y + pt.x * image.rows;
            activeClusters.insert(labels.at<int>(idx, 0));
        }
    }
    
    // Create segmented image showing only active clusters
    cv::Mat segmented = cv::Mat::zeros(image.size(), image.type());
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            int cluster_idx = labels.at<int>(y + x * image.rows, 0);
            if (activeClusters.find(cluster_idx) != activeClusters.end()) {
                segmented.at<cv::Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
                segmented.at<cv::Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
                segmented.at<cv::Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
            }
        }
    }
    
    // Create visualization with original image and segmented parts
    cv::Mat display;
    image.copyTo(display);
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            int cluster_idx = labels.at<int>(y + x * image.rows, 0);
            if (activeClusters.find(cluster_idx) != activeClusters.end()) {
                // Blend original with segmented color
                display.at<cv::Vec3b>(y, x) = 0.7 * image.at<cv::Vec3b>(y, x) + 0.3 * segmented.at<cv::Vec3b>(y, x);
            }
        }
    }
    
    // Mark activation points on the image
    for (const cv::Point& pt : activationPoints) {
        if (pt.x >= 0 && pt.x < image.cols && pt.y >= 0 && pt.y < image.rows) {
            cv::circle(display, pt, 5, cv::Scalar(0, 0, 255), cv::FILLED);
        }
    }
    
    // Display results
    cv::imshow("Original Image", image);
    cv::imshow("Selected Segments", segmented);
    cv::imshow("Highlighted Segments", display);
    cv::waitKey(0);
}



cv::Mat meanShiftSegmentation(const cv::Mat& input_image, 
    double spatial_radius = 10,
    double color_radius = 20,
    int max_level = 2) {
// Validate input
if (input_image.empty()) {
std::cerr << "Error: Input image is empty!" << std::endl;
return cv::Mat();
}

// Convert to LAB color space - better for color-based segmentation
cv::Mat lab_image;
cv::cvtColor(input_image, lab_image, cv::COLOR_BGR2Lab);

// Apply Mean Shift segmentation
cv::Mat segmented;
cv::pyrMeanShiftFiltering(lab_image, segmented, 
    spatial_radius, color_radius, 
    max_level);

// Convert back to BGR color space
cv::Mat result;
cv::cvtColor(segmented, result, cv::COLOR_Lab2BGR);

return result;
}


int main(int argc, char* argv[]) {
    // Loading the image.
    Mat image = imread("../data/004_sugar_box/test_images/4_0001_000121-color.jpg");
    if (image.empty()) {
        cout << "Error: impossible to load image!" << endl;
        return -1;
    }
    std::vector<cv::Point> activationPoints ;
    cv::Point p;
    p.x = 438;
    p.y = 317;
    activationPoints.push_back(p);


    // Apply Mean Shift segmentation
    cv::Mat segmented_image = meanShiftSegmentation(image, 15, 25, 1);

    segmKMeans(segmented_image,activationPoints);
    waitKey(0);
    return 0;
}