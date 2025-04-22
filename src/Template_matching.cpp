#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "../include/graphic.h"

// Configuration constants
constexpr double CONFIDENCE_THRESHOLD = 0.4;
const std::string TEST_IMAGE_PATH = "../data/004_sugar_box/test_images/4_0001_000956-color.jpg";

// Template and mask paths
const std::vector<std::string> MODEL_PATHS = {
    "../data/004_sugar_box/models/view_0_001_color.png",
    // ... [all your other paths]
};

const std::vector<std::string> MASK_PATHS = {
    "../data/004_sugar_box/models/view_0_001_mask.png",
    // ... [all your other masks]
};

struct MatchResult {
    double maxVal;
    cv::Point maxLoc;
    cv::Size size;
    double scale;
    double angle;
    cv::Mat bestTemplate;
};

// Generate a range of values with given step
std::vector<double> generateRange(double min, double max, double step) {
    std::vector<double> values;
    for (double v = min; v <= max; v += step) {
        values.push_back(v);
    }
    return values;
}

// Load and preprocess template with its mask
cv::Mat loadAndProcessTemplate(const std::string& templatePath, 
                              const std::string& maskPath) {
    cv::Mat templateImg = cv::imread(templatePath, cv::IMREAD_COLOR);
    cv::Mat mask = cv::imread(maskPath, cv::IMREAD_GRAYSCALE);

    if (templateImg.empty() || mask.empty()) {
        throw std::runtime_error("Failed to load template or mask: " + templatePath);
    }

    // Binarize mask
    cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);

    // Crop to mask region
    cv::Rect bbox = cv::boundingRect(mask);
    cv::Mat cropped = templateImg(bbox).clone();
    cv::Mat croppedMask = mask(bbox).clone();

    // Apply mask
    cv::Mat mask3ch;
    cv::cvtColor(croppedMask, mask3ch, cv::COLOR_GRAY2BGR);
    cv::bitwise_and(cropped, mask3ch, cropped);

    return cropped;
}

// Perform template matching at different scales and angles
MatchResult matchTemplates(const cv::Mat& testImage, 
                          const std::vector<cv::Mat>& templates,
                          const std::vector<double>& scales,
                          const std::vector<double>& angles) {
    MatchResult bestResult;
    bestResult.maxVal = -1;

    for (const auto& templ : templates) {
        for (double scale : scales) {
            cv::Mat resizedTempl;
            cv::resize(templ, resizedTempl, cv::Size(), scale, scale);

            for (double angle : angles) {
                graphic rotatedTempl(resizedTempl);
                rotatedTempl.rotateImage(angle);
                cv::Mat finalTempl = rotatedTempl.getImage();

                if (testImage.cols < finalTempl.cols || testImage.rows < finalTempl.rows) {
                    continue;
                }

                cv::Mat result;
                cv::matchTemplate(testImage, finalTempl, result, cv::TM_CCOEFF_NORMED);

                double maxVal;
                cv::Point maxLoc;
                cv::minMaxLoc(result, nullptr, &maxVal, nullptr, &maxLoc);

                if (maxVal > bestResult.maxVal) {
                    bestResult.maxVal = maxVal;
                    bestResult.maxLoc = maxLoc;
                    bestResult.size = finalTempl.size();
                    bestResult.scale = scale;
                    bestResult.angle = angle;
                    bestResult.bestTemplate = templ.clone();
                }
            }
        }
    }

    return bestResult;
}

int main() {
    try {
        // Load test image
        cv::Mat testImage = cv::imread(TEST_IMAGE_PATH, cv::IMREAD_COLOR);
        if (testImage.empty()) {
            throw std::runtime_error("Failed to load test image");
        }

        // Generate search parameters
        std::vector<double> scales = generateRange(0.75, 1.5, 0.25);
        std::vector<double> angles = generateRange(-30, 30, 5);

        // Load and preprocess all templates
        std::vector<cv::Mat> templates;
        for (size_t i = 0; i < MODEL_PATHS.size(); i++) {
            templates.push_back(loadAndProcessTemplate(MODEL_PATHS[i], MASK_PATHS[i]));
        }

        // Perform template matching
        MatchResult result = matchTemplates(testImage, templates, scales, angles);

        // Display results
        if (result.maxVal > CONFIDENCE_THRESHOLD) {
            cv::rectangle(testImage, result.maxLoc,
                         cv::Point(result.maxLoc.x + result.size.width,
                                  result.maxLoc.y + result.size.height),
                         cv::Scalar(0, 255, 0), 2);
            std::cout << "Object detected with confidence: " << result.maxVal << std::endl;
        } else {
            std::cout << "No match found (best confidence: " << result.maxVal << ")" << std::endl;
        }

        std::cout << "Best angle: " << result.angle << "°" << std::endl;
        std::cout << "Best scale: " << result.scale << std::endl;

        if (!result.bestTemplate.empty()) {
            cv::imshow("Best Template", result.bestTemplate);
        }
        cv::imshow("Detection Result", testImage);
        cv::waitKey(0);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}