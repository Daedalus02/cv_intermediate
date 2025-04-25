// Include guards.
#ifndef FEATURES_EXTRACTOR_H
#define FEATURES_EXTRACTOR_H

#include <vector>
#include <opencv2/features2d.hpp>

class FeaturesExctractor{

    public:
        // CONSTRUCTORS:
        // This constructor builds the internal extractor.
        FeaturesExctractor() {detector = cv::SIFT::create(0, 3, 0.01, 20, 1.6, true);}

        //FeaturesExctractor(std::string detector_type);

        // FUNCTION MEMBERS:
        // This function get the Mat image where the keypoints and descriptors will be detected, and the adress 
        // of the keypoint vector  and the adress of the descriptor.
        void extract_features(const cv::Mat& image,  std::vector<cv::KeyPoint>& keypoints_models, cv::Mat& descriptors_models){
            detector->detectAndCompute(image, cv::noArray(), keypoints_models, descriptors_models);
        }

    private:
        // DATA MEMBER:
        // HERE WE CAN CHANGE THE TYPE OF THE DETECTOR.
        cv::Ptr<cv::SIFT> detector;

};


#endif
