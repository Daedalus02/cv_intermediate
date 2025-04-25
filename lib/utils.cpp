#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include "../include/utils.h"


// HELPER FUNCTIONS:
    // This function print a rectangle in the given image based on the position 
    // of the keypoints of the matches.
    void draw_box(cv::Mat& image, const std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& keypoints, const cv::Scalar& color){
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
        rectangle(image, top_left, bottom_right, color, 2, cv::LINE_8); 
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
        for (const cv::DMatch& match : matches) {
            cv::Point2f pt = keypoints[match.trainIdx].pt;
            float distance = sqrt(pow(pt.x - point.x, 2) + pow(pt.y - point.y, 2));
            if (distance <= max_distance) {
                filtered_matches.push_back(match);
            }
        }
    }

    // Compute center of mass.
    cv::Point2f compute_com(const std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& keypoints){
        if (matches.empty()) {
            return {-1, -1};
        }

        double total_sum_x = 0;
        double total_sum_y = 0;
        int total_num_points = 0;
        for (const cv::DMatch& match : matches) {
            cv::Point2f pt = keypoints[match.trainIdx].pt;
            total_sum_x += pt.x;
            total_sum_y += pt.y;
            total_num_points++;
        }
    
        cv::Point2f com;
        com.x = total_sum_x / total_num_points;
        com.y = total_sum_y / total_num_points;

        return com;
    }

    // Store in a file names file_name data in format:
    //    <object_id>_<object_name> <xmin> <ymin> <xmax> <ymax>
    void store_label(std::string file_name, std::string object_name, cv::Point min, cv::Point max){

    }


    // Reades all the file names inside the dirctory specified by dir_path and
    // stores all the names inside the vector filenames.
    void get_all_filenames(const std::string& dir_path, std::vector<std::string>& filenames) {
        DIR* dir;
        struct dirent* ent;
        if ((dir = opendir(dir_path.c_str())) != NULL) {
            // process all the files insider the directory
            while ((ent = readdir (dir)) != NULL) {
                std::string file_name = ent->d_name;
                // Don't consider the current directory '.' and the parent ".."
                if (file_name == "." || file_name == "..") {
                    continue;
                }
                if (*(dir_path.end() - 1) == '/') {
                    filenames.push_back(dir_path + file_name);
                } else {
                    filenames.push_back(dir_path + "/" + file_name);
                }
            }
            closedir(dir); // Close the directory.
        }
    }


    // Take as input the command line arguments. The argument of the command line
    // are stored in the strings
    //      pd_dir (power drill models dir path)
    //      mb_dir (mustard bottle models dir path)
    //      sb_dir  (sugar box models dir path)
    //  Function getopt is used to parse the command line.
    void parse_command_line(int argc, char* argv[], std::string& pd_dir, 
        std::string& mb_dir, std::string& sb_dir, std::string& scene) {
    int opt;
    while ((opt = getopt(argc, argv, "s:p:m:i:")) != -1) {
        switch (opt) {
            case 'p':
                pd_dir = optarg;
                break;
            case 'm':
                mb_dir = optarg;
                break;
            case 's':
                sb_dir = optarg;
                break;
            case 'i':
                scene = optarg;
                break;
            case '?':
                std::cerr << "Usage: " << argv[0] << " -p <path> -m <path> -s <path> -i <path>" << std::endl
                        << "  Where:" << std::endl
                        << "    -p is the power drill models dir path" << std::endl
                        << "    -m is the mustard bottle models dir path" << std::endl
                        << "    -s is the sugar box models dir path" << std::endl
                        << "    -i is the input scene image path" << std::endl;
                break;
        }
    }
    }
