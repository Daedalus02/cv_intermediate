
// Include guards.
#ifndef FEATURES_MATCHER_H
#define FEATURES_MATCHER_H

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include <string.h>

using namespace cv;

class FeaturesMatcher{

    public:
        // CONSTRUCTORS:
        // This constructor builds the internal matcher.
        FeaturesMatcher() {matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);}


        // FUNCTION MEMBERS:
        // This function computes the matches and the fill the vector of DMatch based on the 
        // matches from the two descriptors received in arguments.
        void compute_matches(std::vector<std::vector<DMatch>>& matches, const Mat& descriptors1, const Mat& descriptors2){
            matcher->knnMatch(descriptors1, descriptors2, matches, 2);
        }
        

    private:
        // DATA MEMBER:
        // This object is a pointer to a DescriptorMatcher.
        Ptr<DescriptorMatcher> matcher;

};

// HELPER FUNCTIONS:

#endif