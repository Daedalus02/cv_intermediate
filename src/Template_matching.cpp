#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <cmath> 
#include "../include/graphic.h"

namespace fs = std::filesystem;

std::vector<double> generateScales(double minScale, double maxScale, double step) {
    std::vector<double> scales;
    for (double s = minScale; s <= maxScale; s += step) {
        scales.push_back(s);
    }
    return scales;
}

std::vector<double> generateAngles(double minAngle, double maxAngle, double step) {
    std::vector<double> angles;
    for (double a = minAngle; a <= maxAngle; a += step) {
        angles.push_back(a);
    }
    return angles;
}

cv::Mat cropToMask(const cv::Mat& image, const cv::Mat& mask) {
    cv::Rect bbox = cv::boundingRect(mask);
    return image(bbox).clone();
}
   
const std::string models_paths[] = {
    "../data/004_sugar_box/models/view_0_001_color.png",
    "../data/004_sugar_box/models/view_0_002_color.png",
    "../data/004_sugar_box/models/view_0_003_color.png",
    "../data/004_sugar_box/models/view_0_004_color.png",
    "../data/004_sugar_box/models/view_0_005_color.png",
    "../data/004_sugar_box/models/view_0_006_color.png",
    "../data/004_sugar_box/models/view_0_007_color.png",
    "../data/004_sugar_box/models/view_0_008_color.png",
    "../data/004_sugar_box/models/view_0_009_color.png",

    "../data/004_sugar_box/models/view_30_000_color.png",
    "../data/004_sugar_box/models/view_30_001_color.png",
    "../data/004_sugar_box/models/view_30_002_color.png",
    "../data/004_sugar_box/models/view_30_003_color.png",
    "../data/004_sugar_box/models/view_30_004_color.png",
    "../data/004_sugar_box/models/view_30_005_color.png",
    "../data/004_sugar_box/models/view_30_006_color.png",
    "../data/004_sugar_box/models/view_30_007_color.png",
    "../data/004_sugar_box/models/view_30_008_color.png",
    "../data/004_sugar_box/models/view_30_009_color.png",

    "../data/004_sugar_box/models/view_60_000_color.png",
    "../data/004_sugar_box/models/view_60_001_color.png",
    "../data/004_sugar_box/models/view_60_002_color.png",
    "../data/004_sugar_box/models/view_60_003_color.png",
    "../data/004_sugar_box/models/view_60_004_color.png",
    "../data/004_sugar_box/models/view_60_005_color.png",
    "../data/004_sugar_box/models/view_60_006_color.png",
    "../data/004_sugar_box/models/view_60_007_color.png",
    "../data/004_sugar_box/models/view_60_008_color.png",
    "../data/004_sugar_box/models/view_60_009_color.png",
};

const std::string mask_paths[] = {
    "../data/004_sugar_box/models/view_0_001_mask.png",
    "../data/004_sugar_box/models/view_0_002_mask.png",
    "../data/004_sugar_box/models/view_0_003_mask.png",
    "../data/004_sugar_box/models/view_0_004_mask.png",
    "../data/004_sugar_box/models/view_0_005_mask.png",
    "../data/004_sugar_box/models/view_0_006_mask.png",
    "../data/004_sugar_box/models/view_0_007_mask.png",
    "../data/004_sugar_box/models/view_0_008_mask.png",
    "../data/004_sugar_box/models/view_0_009_mask.png",

    "../data/004_sugar_box/models/view_30_000_mask.png",
    "../data/004_sugar_box/models/view_30_001_mask.png",
    "../data/004_sugar_box/models/view_30_002_mask.png",
    "../data/004_sugar_box/models/view_30_003_mask.png",
    "../data/004_sugar_box/models/view_30_004_mask.png",
    "../data/004_sugar_box/models/view_30_005_mask.png",
    "../data/004_sugar_box/models/view_30_006_mask.png",
    "../data/004_sugar_box/models/view_30_007_mask.png",
    "../data/004_sugar_box/models/view_30_008_mask.png",
    "../data/004_sugar_box/models/view_30_009_mask.png",

    "../data/004_sugar_box/models/view_60_000_mask.png",
    "../data/004_sugar_box/models/view_60_001_mask.png",
    "../data/004_sugar_box/models/view_60_002_mask.png",
    "../data/004_sugar_box/models/view_60_003_mask.png",
    "../data/004_sugar_box/models/view_60_004_mask.png",
    "../data/004_sugar_box/models/view_60_005_mask.png",
    "../data/004_sugar_box/models/view_60_006_mask.png",
    "../data/004_sugar_box/models/view_60_007_mask.png",
    "../data/004_sugar_box/models/view_60_008_mask.png",
    "../data/004_sugar_box/models/view_60_009_mask.png",
};

int main() {
    std::string immagineTestPath = "../data/004_sugar_box/test_images/4_0001_000956-color.jpg";
    double confidenceThreshold = 0.4;
    std::vector<double> scales = generateScales(0.75, 1.5, 0.25);
    std::vector<double> angles = generateAngles(-30, 30, 5); 

    cv::Mat imgTestColor = cv::imread(immagineTestPath, cv::IMREAD_COLOR);
    if (imgTestColor.empty()) {
        std::cerr << "Error loading test image." << std::endl;
        return -1;
    }

    double maxValGlobal = -1;
    cv::Point maxLocGlobal;
    cv::Size templateSizeGlobal;
    double best_angle = 90;
    double best_scale = 2;
    cv::Mat best_model = cv::imread(immagineTestPath);

    for (int i = 0; i < 30; i++) {
        cv::Mat templColorOriginal = cv::imread(models_paths[i], cv::IMREAD_COLOR);
        cv::Mat templMask = cv::imread(mask_paths[i], cv::IMREAD_GRAYSCALE);
        
        if (templColorOriginal.empty() || templMask.empty()) {
            std::cerr << "Error loading template or mask: " << models_paths[i] << std::endl;
            continue;
        }

        cv::threshold(templMask, templMask, 127, 255, cv::THRESH_BINARY);
        cv::Mat templCropped = cropToMask(templColorOriginal, templMask);
        cv::Mat maskCropped = cropToMask(templMask, templMask);
        templColorOriginal = templCropped.clone();
        templMask = maskCropped.clone();

        for (double scale : scales) {
            cv::Mat templColorResized;
            cv::Mat maskResized;
            cv::resize(templColorOriginal, templColorResized, cv::Size(), scale, scale);
            cv::resize(templMask, maskResized, cv::Size(), scale, scale, cv::INTER_NEAREST); // Use nearest neighbor for mask
            
            // Ensure mask is still binary after resizing
            cv::threshold(maskResized, maskResized, 127, 255, cv::THRESH_BINARY);

            for (double angle : angles) {
                // Rotate both template and mask
                graphic templRotated(templColorResized);
                graphic maskRotated(maskResized);
                
                templRotated.rotateImage(angle);
                maskRotated.rotateImage(angle);
                
                cv::Mat rotatedTemplate = templRotated.getImage();
                cv::Mat rotatedMask = maskRotated.getImage();
                
                // Ensure mask is still binary after rotation
                cv::threshold(rotatedMask, rotatedMask, 127, 255, cv::THRESH_BINARY);

                // Verify sizes match
                if (rotatedTemplate.size() != rotatedMask.size()) {
                    std::cerr << "Size mismatch after rotation!" << std::endl;
                    continue;
                }

                if (imgTestColor.cols >= rotatedTemplate.cols && imgTestColor.rows >= rotatedTemplate.rows) {
                    cv::Mat result;
                    try {
                        cv::matchTemplate(imgTestColor, rotatedTemplate, result, cv::TM_CCOEFF_NORMED, rotatedMask);
                        
                        double minVal, maxVal;
                        cv::Point minLoc, maxLoc;
                        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

                        if (maxVal > maxValGlobal) {
                            maxValGlobal = maxVal;
                            maxLocGlobal = maxLoc;
                            templateSizeGlobal = rotatedTemplate.size();
                            best_scale = scale;
                            best_angle = angle;
                            best_model = templColorOriginal.clone();
                        }
                    } catch (cv::Exception& e) {
                        std::cerr << "Error in matchTemplate: " << e.what() << std::endl;
                    }
                }
            }
        }
    }

    if (maxValGlobal > confidenceThreshold) {
        cv::Point topLeft = maxLocGlobal;
        cv::Point bottomRight(topLeft.x + templateSizeGlobal.width, topLeft.y + templateSizeGlobal.height);
        cv::rectangle(imgTestColor, topLeft, bottomRight, cv::Scalar(0, 255, 0), 2);
        std::cout << "Object detected with confidence score: " << maxValGlobal << std::endl;
    } else {
        std::cout << "No good match found (threshold: " << confidenceThreshold 
                  << "). Max confidence: " << maxValGlobal << std::endl;
    }

    std::cout << "Best angle: " << best_angle << ", Best scale: " << best_scale << std::endl;
    cv::imshow("Best template", best_model);
    cv::imshow("Detection Result", imgTestColor);
    cv::waitKey(0);

    return 0;
}