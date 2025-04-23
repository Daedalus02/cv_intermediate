#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;


void segmKMeans(Mat& image) {
    Mat samples = Mat(image.rows * image.cols, 3, CV_32F);
    
    // Prepare data for k-means.
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            for (int z = 0; z < 3; z++) {
                samples.at<float>(y + x * image.rows, z) = image.at<Vec3b>(y, x)[z];
            }
        }
    }
    
    // Number of clusters.
    int clusterCount = 15;
    Mat labels;
    int attempts = 5;
    Mat centers;
    
    // Using k-means.
    kmeans(samples, clusterCount, labels, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
    
    // Rebuilding the segmented image.
    Mat segmented = Mat(image.size(), image.type());
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            int cluster_idx = labels.at<int>(y + x * image.rows, 0);
            segmented.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
            segmented.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
            segmented.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
        }
    }
    
    // Dysplaying results.
    imshow("Original image", image);
    imshow("K-Means segmented", segmented);
    waitKey(0);
}


int main(int argc, char* argv[]) {
    // Loading the image.
    Mat image = imread("../data/004_sugar_box/test_images/4_0001_000121-color.jpg");
    if (image.empty()) {
        cout << "Error: impossible to load image!" << endl;
        return -1;
    }
    
    segmKMeans(image);
    
    destroyAllWindows();
    return 0;
}