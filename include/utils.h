#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core/types.hpp>
#include <utility>
#include <vector>

// Draw a box in the input image. The box is computed starting from the detected
// keypoints. The drawn box contains all the detected keypoints.
// Returns the top left corner and the bottom right corner of the box.
std::pair<cv::Point2i, cv::Point2i> draw_box(cv::Mat& image, const std::vector<cv::Point2i>& points, 
        const std::vector<cv::KeyPoint>& keypoints, const cv::Scalar& color);
 

// Filtering with Lowe filter.
void lowe_filter(const std::vector<std::vector<cv::DMatch>>& matches,  float threshold, 
        std::vector<cv::DMatch>& good_matches); 

// Filtering using max distance from center of mass.
void max_distance_filter(float max_distance, const std::vector<cv::Point2i>& points, 
        cv::Point2f center, std::vector<cv::Point2i>& filtered_points);

// Compute the vector or points that have at least min_match points within a window of ray max kernel_size.
void kernel_filter(int max_kernel_size, int min_match,
        const std::vector<cv::Point2i>&  filtered_points, 
        std::vector<cv::Point2i>& final_points);

// Compute center of mass.
cv::Point2d compute_com(const std::vector<cv::Point2i>& points, const std::vector<cv::KeyPoint>& keypoints);

// Store in a file names file_name data in format:
//    <object_id>_<object_name> <xmin> <ymin> <xmax> <ymax>
void store_label(const std::string& file_name, const std::string& object_name, const cv::Point& min, const cv::Point& max);

// Reades all the file names inside the dirctory specified by dir_path and
// stores all the names inside the vector filenames.
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
