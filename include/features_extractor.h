// Authors: Chinello Alessandro, Piai Luca, Scantamburlo Mattia
// (Read the report)

#ifndef FEATURES_EXTRACTOR_H
#define FEATURES_EXTRACTOR_H

#include <vector>
#include <opencv2/features2d.hpp>

class FeaturesExctractor{

    public:
        // This constructor builds the internal extractor.
        // Under the hood it uses SIFT.
        FeaturesExctractor() {
            // The parameters was determined empirically.
            detector = cv::SIFT::create(0, 3, 0.03, 20, 1.6, true);
        }

        // This function get the Mat image where the keypoints 
        // and descriptors will be detected, they will be stored 
        // respectively in 'keypoints' and 'descriptors' vectors.
        void extract_features(const cv::Mat& image, 
                std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
            detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
        }

    private:
        cv::Ptr<cv::SIFT> detector;
};

#endif
