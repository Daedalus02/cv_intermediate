#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include <string.h>
#include "../include/utils.h"


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
        for (size_t i = 0; i < matches.size(); i++) {
            if (matches[i][0].distance < threshold * matches[i][1].distance) {
                good_matches.push_back(matches[i][0]);
            }
        }
    }

    // Filtering using max distance from center of mass.
    void max_distance_filter(float max_distance, std::vector<cv::DMatch>& matches, 
        std::vector<cv::KeyPoint>& keypoints, cv::Point2f point, std::vector<cv::DMatch>& filtered_matches){
        double total_sum_x = 0;
        double total_sum_y = 0;
        int total_num_points = 0;
        for (const auto& match : matches) {
            cv::Point2f pt = keypoints[match.trainIdx].pt;
            float distance = sqrt(pow(pt.x - point.x, 2) + pow(pt.y - point.y, 2));
            if (distance <= max_distance) {
                filtered_matches.push_back(match);
                total_sum_x += pt.x;
                total_sum_y += pt.y;
                total_num_points++;
            }
        }
    }

    // Compute center of mass.
    cv::Point2f compute_com(std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& keypoints){
        double total_sum_x = 0;
        double total_sum_y = 0;
        int total_num_points = 0;
        for (const auto& match : matches) {
            cv::Point2f pt = keypoints[match.trainIdx].pt;
            total_sum_x += pt.x;
            total_sum_y += pt.y;
            total_num_points++;
        }
    
        cv::Point2f com;
        if (total_num_points > 0) {
            com.x = total_sum_x / total_num_points;
            com.y = total_sum_y / total_num_points;
            //std::cout << "Total Center of Mass: (" << com.x << ", " << com.y << ")" << std::endl;
        }
        else {
            //std::cout << "No good matches found at all." << std::endl;
            com.x = fake_com.x;
            com.y = fake_com.y;
        }
        return com;
    }

    // Store in a file names file_name data in format:
    //    <object_id>_<object_name> <xmin> <ymin> <xmax> <ymax>
    void store_label(std::string file_name, std::string object_name, cv::Point min, cv::Point max){

    }
