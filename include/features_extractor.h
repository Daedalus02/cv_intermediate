// Include guards.
#ifndef FEATURES_EXTRACTOR_H
#define FEATURES_EXTRACTOR_H

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include <string.h>

using namespace cv;

class features_exctractor{

    public:
        // CONSTRUCTORS:
        // This constructor builds the internal extractor.
        features_exctractor() {detector = SIFT::create();}

        features_exctractor(std::string detector_type);

        // FUNCTION MEMBERS:
        // This function get the Mat image where the keypoints and descriptors will be detected, and the adress 
        // of the keypoint vector  and the adress of the descriptor.
        void extract_features(const Mat& image,  std::vector<KeyPoint>& keypoints_models, Mat& descriptors_models){
            detector->detectAndCompute(image, noArray(), keypoints_models, descriptors_models);
        }

    private:
        // DATA MEMBER:
        // HERE WE CAN CHANGE THE TYPE OF THE DETECTOR.
        Ptr<SIFT> detector;

};

// HELPER FUNCTIONS:

#endif