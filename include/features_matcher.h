// Authors: Chinello Alessandro, Piai Luca, Scantamburlo Mattia
// (Read the report)

#ifndef FEATURES_MATCHER_H
#define FEATURES_MATCHER_H

#include <opencv2/features2d.hpp>

class FeaturesMatcher{

    public:
        // This constructor builds the internal matcher.
        // Under the hood it uses FLANN matcher.
        FeaturesMatcher() {
            matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        }

        // This function computes the matches and fill the vector 'matches'.
        void compute_matches(std::vector<std::vector<cv::DMatch>>& matches,
                const cv::Mat& descriptors1, const cv::Mat& descriptors2) {
            matcher->knnMatch(descriptors1, descriptors2, matches, 2);
        }

    private:
        // This object is a pointer to a DescriptorMatcher.
        cv::Ptr<cv::DescriptorMatcher> matcher;
};

#endif
