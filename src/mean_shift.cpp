#include <opencv2/opencv.hpp>
#include <iostream>


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

void displayComparison(const cv::Mat& original, 
                      const cv::Mat& segmented,
                      const std::string& window_name = "Mean Shift Result") {
    // Create a combined image for display
    cv::Mat combined(original.rows, original.cols * 2, original.type());
    
    original.copyTo(combined(cv::Rect(0, 0, original.cols, original.rows)));
    segmented.copyTo(combined(cv::Rect(original.cols, 0, original.cols, original.rows)));
    
    // Add labels
    cv::putText(combined, "Original", cv::Point(20, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined, "Segmented", cv::Point(original.cols + 20, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    
    cv::imshow(window_name, combined);
}

int main(int argc, char** argv) {
    // Load image from command line or use default
    cv::Mat input_image = cv::imread("../data/004_sugar_box/test_images/4_0001_000121-color.jpg");
    if (input_image.empty()) {
        std::cout << "Error: impossible to load image!" << std::endl;
        return -1;
    }

    // Apply Mean Shift segmentation
    cv::Mat segmented_image = meanShiftSegmentation(input_image, 15, 25, 1);

    // Display results
    displayComparison(input_image, segmented_image);
    
    // Wait for key press
    cv::waitKey(0);
    
    return 0;
}