// Authors: Chinello Alessandro, Piai Luca, Scantamburlo Mattia
// (Read the report)

#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core/types.hpp>
#include <utility>
#include <vector>

// Draw a box in the input image. The box is computed starting from the detected
// keypoints. The drawn box contains all the detected keypoints.
// Returns the top left corner and the bottom right corner of the box.
std::pair<cv::Point2i, cv::Point2i> bounding_box_coord(const cv::Mat& image, 
        const std::vector<cv::Point2i>& points, const std::vector<cv::KeyPoint>& keypoints, double expansion);

// Filtering with Lowe filter.
void lowe_filter(const std::vector<std::vector<cv::DMatch>>& matches, float threshold, 
        std::vector<cv::DMatch>& good_matches); 

// Filtering using max distance from 'center'.
// All the points in 'points' that have a distance bigger than 'max_distance'
// from 'center' are not added to 'filtered_points'.
void max_distance_filter(float max_distance, const std::vector<cv::Point2i>& points,
        cv::Point2f center, std::vector<cv::Point2i>& filtered_points);

// For each point x in 'points' compute how many points have a distance
// that is lower than 'max_distance'.
// If point x has less than 'min_neighbors' neighbors, then it is
// not added to filtered_points.
void neighbor_filter(int max_distance, int min_neighbors,
        const std::vector<cv::Point2i>& points, 
        std::vector<cv::Point2i>& filtered_points);

// Compute center of mass of the points in 'points'.
cv::Point2d compute_com(const std::vector<cv::Point2i>& points);

// Store the lablel in a file called 'file_name'.
// The format of the label is:
//    <object_id>_<object_name> <xmin> <ymin> <xmax> <ymax>
void store_label(const std::string& file_name, const std::string& object_name,
        const cv::Point2i& min, const cv::Point2i& max);

// Reades all the file names inside the dirctory specified by 'dir_path' and
// stores all the names in 'filenames'.
void get_all_filenames(const std::string& dir_path, std::vector<std::string>& filenames);

// Take as input the command line arguments. The argument of the command line
// are stored in the strings
//      pd_dir (power drill models dir path)
//      mb_dir (mustard bottle models dir path)
//      sb_dir  (sugar box models dir path)
//  Function getopt is used to parse the command line.
void parse_command_line(int argc, char* argv[], std::string& pd_dir, 
        std::string& mb_dir, std::string& sb_dir, std::string& scene,
        std::string& label);

#endif
