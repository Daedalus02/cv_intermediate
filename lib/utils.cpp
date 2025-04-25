#include <fstream>
#include <iostream>
#include <unistd.h>
#include <dirent.h>
#include <opencv2/imgproc.hpp>

#include "../include/utils.h"

std::pair<cv::Point2i, cv::Point2i> draw_box(cv::Mat& image, const std::vector<cv::Point2i>& points, const std::vector<cv::KeyPoint>& keypoints, const cv::Scalar& color){
    cv::Point2i top_left(image.rows, image.cols);
    cv::Point2i bottom_right(0, 0);
    for (const cv::Point2i& pt : points) {
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
    return {top_left, bottom_right};
}

void lowe_filter(const std::vector<std::vector<cv::DMatch>>& matches,  float threshold, 
        std::vector<cv::DMatch>& good_matches){
    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < threshold * matches[i][1].distance) {
            good_matches.push_back(matches[i][0]);
        }
    }
}

void max_distance_filter(float max_distance, const std::vector<cv::Point2i>& points, cv::Point2f center, std::vector<cv::Point2i>& filtered_points){
    for (const cv::Point2i& pt : points) {
        float distance = sqrt(pow(pt.x - center.x, 2) + pow(pt.y - center.y, 2));
        if (distance <= max_distance) {
            filtered_points.push_back(pt);
        }
    }
}


void kernel_filter(int max_kernel_size, int min_match, 
    const std::vector<cv::Point2i>&  points, 
    std::vector<cv::Point2i>& final_points){
    final_points = {};
    for(const auto& pt : points){
        std::vector<cv::Point2i> kernel_points;
        max_distance_filter(max_kernel_size, points, pt, kernel_points);
        if(kernel_points.size()> min_match){
            final_points.push_back(pt);
        }
    }   
}




cv::Point2d compute_com(const std::vector<cv::Point2i>& points, const std::vector<cv::KeyPoint>& keypoints){
    if (points.empty()) {
        return {-1, -1};
    }

    double total_sum_x = 0;
    double total_sum_y = 0;
    int total_num_points = 0;
    for (const cv::Point2i& pt : points) {
        total_sum_x += pt.x;
        total_sum_y += pt.y;
        total_num_points++;
    }

    cv::Point2d com;
    com.x = total_sum_x / total_num_points;
    com.y = total_sum_y / total_num_points;

    return com;
}

void store_label(const std::string& file_name, const std::string& object_name, const cv::Point2i& min, const cv::Point2i& max){
    std::ofstream outfile;
    outfile.open(file_name, std::ios_base::app);

    if (outfile.is_open()) {
        outfile << object_name << " " << min.x << " " << min.y << " " << max.x << " " << max.y << std::endl;
        outfile.close();
    } else {
        std::cerr << "Unable to open file: " << file_name << std::endl;
    }
}

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

void parse_command_line(int argc, char* argv[], std::string& pd_dir, 
        std::string& mb_dir, std::string& sb_dir, std::string& scene,
        std::string& label) {
    int opt;
    while ((opt = getopt(argc, argv, "s:p:m:i:l:")) != -1) {
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
            case 'l':
                label = optarg;
                break;
            case '?':
                std::cerr << "Usage: " << argv[0] << " -p <path> -m <path> -s <path> -i <path> -l <path>" << std::endl
                    << "  Where:" << std::endl
                    << "    -p is the power drill models dir path" << std::endl
                    << "    -m is the mustard bottle models dir path" << std::endl
                    << "    -s is the sugar box models dir path" << std::endl
                    << "    -i is the input scene image path" << std::endl
                    << "    -l is the label path associated with the scene" << std::endl;
                break;
        }
    }
}
