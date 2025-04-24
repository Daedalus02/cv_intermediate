/*
 * Approach used:
 *      1) Compute feature matching between all the 30 rgb models and the scene image.
 *      2) Filter the matches found in step (1) with Lowes ratio test.
 *      3) Compute the center of mass (COM) of the matches located in the scene image (green circles).
 *      4) Neglect all the matches that are far away from the COM.
 *      5) Update the position of the COM of the matches, then draw it in red.
 */

 #include <iostream>
 #include "opencv2/core.hpp"
 #include "opencv2/core/types.hpp"
 #include "opencv2/imgcodecs.hpp"
 #include "opencv2/highgui.hpp"
 #include "opencv2/features2d.hpp"
 #include "opencv2/imgproc.hpp"

#include <unistd.h>
#include <string>
#include <dirent.h>
 
using namespace cv;
using std::cout;
using std::endl;

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
                std::cerr << "Usage: " << argv[0] << " -p <path> -m <path> -s <path> -i <path>" << endl
                          << "  Where:" << endl
                          << "    -p is the power drill models dir path" << endl
                          << "    -m is the mustard bottle models dir path" << endl
                          << "    -s is the sugar box models dir path" << endl
                          << "    -i is the input scene image path" << endl;
                break;
        }
    }
}
 
 int main(int argc, char* argv[]) {

     /// Inputs are:
     ///    - scene
     ///    - models dir for sugar box
     ///    - models dir for power drill
     ///    - models dir for mustard bottle
     ///

     // Get the directories paths of the models.
    std::string scene_image_path{};
    std::string pd_models_dirpath{}; // Power drill models dir path
    std::string mb_models_dirpath{}; // Mustar bottle models dir path
    std::string sb_models_dirpath{}; // Sugar box models dir path
    parse_command_line(argc, argv, pd_models_dirpath, mb_models_dirpath, sb_models_dirpath, scene_image_path);

    if (pd_models_dirpath.empty() || mb_models_dirpath.empty() 
            || sb_models_dirpath.empty() || scene_image_path.empty()) {
        std::cerr << "Error in parsing the command line... aborting.\n";
        return 1;
    }

    //cout << "Models directories:" << endl;
    //cout << pd_models_dirpath << endl
    //     << mb_models_dirpath << endl
    //     << sb_models_dirpath << endl << endl;


    // Get the path of each models inside the speficied directories.
    std::vector<std::string> pd_models_images_paths; // Path of each power drill model.
    std::vector<std::string> mb_models_images_paths; // Path of each mustard bottle model.
    std::vector<std::string> sb_models_images_paths; // Path of each sugar box model.

    get_all_filenames(pd_models_dirpath, pd_models_images_paths);
    get_all_filenames(mb_models_dirpath, mb_models_images_paths);
    get_all_filenames(sb_models_dirpath, sb_models_images_paths);

    if (pd_models_images_paths.empty() || mb_models_images_paths.empty() || sb_models_images_paths.empty()) {
        std::cerr << "Error: one of the models directory is empty... aborting\n";
    }

    //cout << "pd_models_images_paths:\n";
    //for (auto const& s : pd_models_images_paths)
    //    cout << s << endl;
    //cout << pd_models_images_paths.size() << endl << endl;

    //cout << "mb_models_images_paths\n";
    //for (auto const& s : mb_models_images_paths)
    //    cout << s << endl;
    //cout << mb_models_images_paths.size() << endl << endl;

    //cout << "sb_models_images_paths\n";
    //for (auto const& s : sb_models_images_paths)
    //    cout << s << endl;
    //cout << sb_models_images_paths.size() << endl << endl;

 
     // Define the models images.
     std::vector<Mat> models;
     for (const auto& p : pd_models_images_paths) {
         models.push_back(imread(p, IMREAD_GRAYSCALE));
     }
 
     // Define the scene image.
     Mat scene = imread(scene_image_path, IMREAD_GRAYSCALE);
     //blur(scene, scene, Size(3, 3));
 
     //-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors.
 
     // Define the SIFT detector.
     Ptr<SIFT> detector = SIFT::create();
 
     // Keypoints and descriptor of each model.
     std::vector<std::vector<KeyPoint>> keypoints_models(models.size());
     std::vector<Mat> descriptors_models(models.size());
 
     // Keypoints and descriptor of the scene.
     std::vector<KeyPoint> keypoints_scene;
     Mat descriptors_scene;
 
     // Compute keypoints and descriptors.
     for (int i = 0; i < models.size(); i++) {
         detector->detectAndCompute(models[i], noArray(), keypoints_models[i], descriptors_models[i]);
     }
     detector->detectAndCompute(scene, noArray(), keypoints_scene, descriptors_scene);
 
     //-- Step 2: Matching descriptor vectors with a BRUTEFORCE matcher
 
     // Define the Brute force matcher.
     Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
 
     // Vector that stores the good matches.
     //std::vector<std::vector< std::vector<DMatch> >> knn_matches(models.size());
     std::vector<std::vector<DMatch>> knn_matches;
 
     // Compute the matches between the scene and each model.
     for (int i = 0; i < models.size(); i++) {
         std::vector<std::vector<DMatch>> tmp;
         matcher->knnMatch(descriptors_models[i], descriptors_scene, tmp, 2);
         knn_matches.insert(knn_matches.end(), tmp.begin(), tmp.end());
     }
 
     // Filter matches using the Lowe's ratio test.
     const float ratio_thresh = 0.85;
     std::vector<DMatch> good_matches;
     for (size_t i = 0; i < knn_matches.size(); i++) {
         if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
             good_matches.push_back(knn_matches[i][0]);
         }
     }
 
     // Calculate the total center of mass from all good matches
     double total_sum_x = 0;
     double total_sum_y = 0;
     int total_num_points = 0;
     for (const auto& match : good_matches) {
         Point2f pt = keypoints_scene[match.trainIdx].pt;
         total_sum_x += pt.x;
         total_sum_y += pt.y;
         total_num_points++;
     }
 
     Point2f com;
     if (total_num_points > 0) {
         com.x = total_sum_x / total_num_points;
         com.y = total_sum_y / total_num_points;
         cout << "Total Center of Mass: (" << com.x << ", " << com.y << ")" << endl;
     }
     else {
         cout << "No good matches found at all." << endl;
         com.x = -1;
         com.y = -1;
     }
 
     //-- Step 3: Filter matches based on distance to the center of mass
     float max_distance_threshold = 120; // Adjust this threshold as needed
     std::vector<DMatch> filtered_matches;;
 
     total_sum_x = 0;
     total_sum_y = 0;
     total_num_points = 0;
     for (const auto& match : good_matches) {
         Point2f pt = keypoints_scene[match.trainIdx].pt;
         float distance = sqrt(pow(pt.x - com.x, 2) + pow(pt.y - com.y, 2));
         if (distance <= max_distance_threshold) {
             filtered_matches.push_back(match);
             total_sum_x += pt.x;
             total_sum_y += pt.y;
             total_num_points++;
         }
     }
 
     // Recalculate the center of mass with the filtered matches
     Point2f new_com;
     if (total_num_points > 0) {
         new_com.x = total_sum_x / total_num_points;
         new_com.y = total_sum_y / total_num_points;
         cout << "Filtered Center of Mass: (" << new_com.x << ", " << new_com.y << ")" << endl;
     }
     else {
         cout << "No matches survived the filtering." << endl;
         new_com = com; // Keep the old COM or set to a default value
     }
 
     // Visualize the center of mass.
     Mat scene_with_centers;
     cvtColor(scene, scene_with_centers, COLOR_GRAY2BGR);
     Scalar color_center = Scalar(0, 0, 255);
     int radius_center = 5;
     int thickness_center = 2;
     if (total_num_points > 0)
         circle(scene_with_centers, new_com, radius_center, color_center, thickness_center);
     imshow("Scene with Filtered Center of Mass", scene_with_centers);
 
 
     // -- Draw the filtered matches
     Mat scene_matches;
     cvtColor(scene, scene_matches, COLOR_GRAY2BGR);
     Scalar color = Scalar(0, 255, 0);
     int radius = 3;
     int thickness = 2;
     for (const auto& match : filtered_matches) {
         Point2f pt2 = keypoints_scene[match.trainIdx].pt;
         circle(scene_matches, pt2, radius, color, thickness);
     }
 
     // Last step draw the rectangle.
     Point2f topLeft(scene.rows, scene.cols);
     Point2f bottomRight(0, 0);
     for (const auto& match : filtered_matches) {
         Point2f pt = keypoints_scene[match.trainIdx].pt;
         if (pt.y < topLeft.y)
             topLeft.y = pt.y;
         if (pt.x < topLeft.x)
             topLeft.x = pt.x;
         if (pt.y > bottomRight.y)
             bottomRight.y = pt.y;
         if (pt.x > bottomRight.x)
             bottomRight.x = pt.x;
     }
     rectangle(scene_matches, topLeft, bottomRight, Scalar(255, 0, 0), 2, LINE_8); 
 
     cout << "TOT FILTERED MATCHES: " << filtered_matches.size() << endl;

     imshow("Filtered Matches on Scene", scene_matches);
     waitKey();
     return 0;
}
