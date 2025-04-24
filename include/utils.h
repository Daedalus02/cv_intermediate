
// Include guards.
#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include <string.h>



// GLOBAL CONSTANTS
const cv::Point2f fake_com (-1,-1);

// HELPER FUNCTIONS:
    //
    void draw_box(cv::Mat& image, const std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& keypoints);

    // Filtering with Lowe filter.
    void lowe_filter(std::vector<std::vector<cv::DMatch>>& matches,  float threshold, 
        std::vector<cv::DMatch>& good_matches); 

    // Filtering using max distance from center of mass.
    void max_distance_filter(float max_distance, std::vector<cv::DMatch>& matches, 
        std::vector<cv::KeyPoint>& keypoints, cv::Point2f point, std::vector<cv::DMatch>& filtered_matches);

    // Compute center of mass.
    cv::Point2f compute_com(std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& keypoints);

    // Store in a file names file_name data in format:
    //    <object_id>_<object_name> <xmin> <ymin> <xmax> <ymax>
    void store_label(std::string file_name, std::string object_name, cv::Point min, cv::Point max);

#endif