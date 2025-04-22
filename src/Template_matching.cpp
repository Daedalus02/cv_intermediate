#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <cmath> 
# include "../include/graphic.h"

// Declaring the namespace with the filesystem of the current OS.
namespace fs = std::filesystem;

// Generate scales between minScale and maxScale with step equal to step.
std::vector<double> generateScales(double minScale, double maxScale, double step) {
    std::vector<double> scales;
    for (double s = minScale; s <= maxScale; s += step) {
        scales.push_back(s);
    }
    return scales;
}

// Generate angles between minAngle and maxAngle with step equal to step.
std::vector<double> generateAngles(double minAngle, double maxAngle, double step) {
    std::vector<double> angles;
    for (double a = minAngle; a <= maxAngle; a += step) {
        angles.push_back(a);
    }
    return angles;
}


const   std::string models_paths[] = {
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

    // Defining the path of the image to test the algorithm on.
    std::string immagineTestPath = "../data/004_sugar_box/test_images/4_0049_000003-color.jpg";
    // Defining the path of the folder with all the models images.
    std::string modelsFolderPath = "../data/004_sugar_box/models/";

    // Defining the confidence threshold for the test.
    double confidenceThreshold = 0.4;
    // Defining a vector of double values for all the scales to test the template on.
    std::vector<double> scales = generateScales(0.75, 1.5, 0.25);
    // Defining a vector of double values for all the inclinations to test the template on.
    std::vector<double> angles = generateAngles(-30, 30, 5); 

    // Loading the image in both color scale and grey scale to test both.
    cv::Mat imgTestColor = cv::imread(immagineTestPath, cv::IMREAD_COLOR);
    cv::Mat imgTestGray = cv::imread(immagineTestPath, cv::IMREAD_GRAYSCALE);

    // Testing the validity of both the Mat objects.
    if (imgTestColor.empty() || imgTestGray.empty()) {
        // If one of the images was not present in the path, print error message and quit.
        std::cerr << "Errore nel caricamento dell'immagine di test." << std::endl;
        return -1;
    }


    double maxValGlobal = -1;
    // Point object to store the point of higher probability for a match. 
    cv::Point maxLocGlobal;
    // Size object for storing the size of the maxLocGlobal point. 
    cv::Size templateSizeGlobal;

    double best_angle = 90;
    double best_scale = 2;
    cv::Mat best_model = cv::imread(immagineTestPath);


    // Iterating through all the possible models images in the directory.
    for (int i = 0; i < size(models_paths); i++) {

        // Creating a variable of type Mat with the loaded model image in grayscale.
        cv::Mat templGrayOriginal = cv::imread(models_paths[i], cv::IMREAD_GRAYSCALE);
        cv::Mat templMask = cv::imread(mask_paths[i], cv::IMREAD_GRAYSCALE);
        // Ensure mask is binary (0 or 255)
        cv::threshold(templMask, templMask, 127, 255, cv::THRESH_BINARY);

        cv::bitwise_and(templGrayOriginal, templMask, templGrayOriginal);
        //cv::imshow(models_paths[i], templGrayOriginal);

        // Checking if the loaded model image is valid.
        if (!templGrayOriginal.empty()) {
            // Iterating over all the possible scales.
            for (double scale : scales) {
                // Resizing the template image with current scale configuration.
                cv::Mat templGrayResized;
                cv::resize(templGrayOriginal, templGrayResized, cv::Size(), scale, scale);

                // Combining all the possible angles for the rotation of the template.
                for (double angle : angles) {
                    // Rotating the current scale image.
                    graphic templGrayRotated = graphic(templGrayResized);
                    templGrayRotated.rotateImage(angle);

                    // Checking if the current test image is bigger in width and height than the template.
                    if (imgTestGray.cols >= templGrayRotated.cols() && imgTestGray.rows >= templGrayRotated.rows()) {

                        // Storing the result if the template sliding process in "result" variable.
                        cv::Mat result;
                        // Using the TM_CCOEFF_NORMED method.
                        cv::matchTemplate(imgTestGray, templGrayRotated.getImage(), result, cv::TM_CCOEFF_NORMED);
                        // Storing the mininmum and maximum values and location among all the possible positions.
                        double minVal; double maxVal;
                        cv::Point minLoc; cv::Point maxLoc;
                        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

                        // Checking if the maximum value found in the current configuration is better than the global one.
                        if (maxVal > maxValGlobal) {
                            // Fixing the new global maximum value amd position.
                            maxValGlobal = maxVal;
                            maxLocGlobal = cv::Point(cvRound(maxLoc.x), cvRound(maxLoc.y));
                            // Storing the size of the current template.
                            templateSizeGlobal = cv::Size(templGrayRotated.cols(), templGrayRotated.rows());
                            best_scale = scale;
                            best_angle = angle;
                            best_model = templGrayOriginal.clone();
                        }
                    }
                }
            }
        }
    
    }

    // Checking whether the maximum value found among all the configuration is good enough.
    if (maxValGlobal > confidenceThreshold) {
        // Painting the rectangle around the position of higher matching value. 
        cv::Point topLeft = maxLocGlobal;
        cv::Point bottomRight(topLeft.x + templateSizeGlobal.width, topLeft.y + templateSizeGlobal.height);
        cv::rectangle(imgTestColor, topLeft, bottomRight, cv::Scalar(0, 255, 0), 2);
        std::cout << "Object detected with confidence score: " << maxValGlobal << std::endl;
    } else {
        std::cout << "No suffuciently good corrispondence found among all templates (threshold: " << confidenceThreshold << "). Max confidence: " << maxValGlobal << std::endl;
    }

    std::cout<<"Best angle is: "<<best_angle<<std::endl;
    std::cout<<"Best scale is: "<<best_scale<<std::endl;
    cv::namedWindow("Best template");
    cv::imshow("Best template", best_model);
    cv::imshow("Image with detected object", imgTestColor);
    cv::waitKey(0);

    return 0;
}