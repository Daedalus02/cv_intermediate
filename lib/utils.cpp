#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include <string.h>
#include "../include/utils.h"



// GLOBAL CONSTANTS

// HELPER FUNCTIONS:
    // This function print a rectangle in the given image based on the position 
    // of the keypoints of the matches.
    void draw_box(cv::Mat& image, const std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& keypoints){
        cv::Point2f top_left(image.rows, image.cols);
        cv::Point2f bottom_right(0, 0);
        for (const auto& match : matches) {
            cv::Point2f pt = keypoints[match.trainIdx].pt;
            if (pt.y < top_left.y)
                top_left.y = pt.y;
            if (pt.x < top_left.x)
                top_left.x = pt.x;
            if (pt.y > bottom_right.y)
                bottom_right.y = pt.y;
            if (pt.x > bottom_right.x)
                bottom_right.x = pt.x;
        }
        rectangle(image, top_left, bottom_right, cv::Scalar(255, 0, 0), 2, cv::LINE_8); 
    }

    // Filtering with Lowe filter.
    void lowe_filter(std::vector<std::vector<cv::DMatch>>& matches,  float threshold, 
        std::vector<cv::DMatch>& good_matches){

        }

    // Filtering using max distance from center of mass.
    void max_distance_filter(int max_distance, std::vector<cv::DMatch>& matches, 
        std::vector<cv::KeyPoint>& keypoints, cv::Point2f point){

        }

    // Compute center of mass.
    cv::Point2f compute_com(std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& keypoints){

    }

    // Store in a file names file_name data in format:
    //    <object_id>_<object_name> <xmin> <ymin> <xmax> <ymax>
    void store_label(std::string file_name, std::string object_name, cv::Point min, cv::Point max){

    }
