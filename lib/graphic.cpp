#include "../include/graphic.h"
#include <stdexcept>

    // Constructor with argument image.
    graphic::graphic(const cv::Mat& source){
        if(source.empty()){
            throw std::invalid_argument("The Mat object passed to the image was not valid!");
        }
        graphic::image = source.clone();
    }

    // This function is used to compute all the possible rotation of the template. 
    void graphic::rotateImage(double angle) {
        cv::Point2f center(graphic::image.cols / 2.0f, graphic::image.rows / 2.0f);
        cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::warpAffine(graphic::image, graphic::image, rotMat, graphic::image.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    }

    // Setter function.
    void graphic::setImage(const cv::Mat& source){
        if(source.empty()){
            throw std::invalid_argument("The Mat object passed to the image was not valid!");
        }
        graphic::image = source.clone();
    }