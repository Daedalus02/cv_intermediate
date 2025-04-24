
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

// HELPER FUNCTIONS:
    //
    void draw_box(cv::Mat& image);

    // Filtering with Lowe filter.
    void lowe_filter(std::vector<std::vector<DMatch>>& matches,  float threshold, std::vector<DMatch>& good_matches); 

    // Filtering using max distance from center of mass.
    void max_distance_filter(int max_distance, std::vector<DMatch>& matches, std::vector<KeyPoint>& keypoints, Point2f point);

    // Compute center of mass.
    Point2f compute_com(std::vector<DMatch>& matches, std::vector<KeyPoint>& keypoints);

    // Store in a file names file_name data in format:
    //    <object_id>_<object_name> <xmin> <ymin> <xmax> <ymax>
    void store_label(std::string file_name, std::string object_name, cv::Point min, cv::Point max);

#endif