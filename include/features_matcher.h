// Include guards.
#ifndef FEATURES_MATCHER_H
#define FEATURES_MATCHER_H

#include <opencv2/features2d.hpp>

class FeaturesMatcher{

    public:
        // CONSTRUCTORS:
        // This constructor builds the internal matcher.
        FeaturesMatcher() {matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);}


        // FUNCTION MEMBERS:
        // This function computes the matches and the fill the vector of DMatch based on the 
        // matches from the two descriptors received in arguments.
        void compute_matches(std::vector<std::vector<cv::DMatch>>& matches, const cv::Mat& descriptors1, const cv::Mat& descriptors2){
            matcher->knnMatch(descriptors1, descriptors2, matches, 2);
        }
        

    private:
        // DATA MEMBER:
        // This object is a pointer to a DescriptorMatcher.
        cv::Ptr<cv::DescriptorMatcher> matcher;

};

// HELPER FUNCTIONS:

#endif
