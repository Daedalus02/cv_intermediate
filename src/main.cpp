// Authors: Chinello Alessandro, Piai Luca, Scantamburlo Mattia
// (Read the report)

#include <iostream>
#include <map>

#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../include/utils.h"
#include "../include/features_extractor.h"
#include "../include/features_matcher.h"
#include "../include/performance_metrics.h"


// Used to normalize the density of matches inside the colored box.
const double scale_factor = 1000;

// Parameters associated to each object class.
// The value below are not magic numbers, they were determinated
// during the tuning phase.
//
// The values of the vectors below are used for: 
//      - Lowe's threshold value, 
//      - Distance from center of mass, 
//      - Ray of the min number of point in ray, 
//      - Min number of point in ray,
//      - Density of points in the rectangle (multiplied by scale factor),
//      - Min number of matches to consider the object detected. 
const std::vector<float> pd_params = {0.8, 152, 50, 10, 1.0, 35};
const std::vector<float> mb_params = {0.8, 150, 80, 15, 0.8, 50};
const std::vector<float> sb_params = {0.75, 160, 80, 20, 1.25, 40};

// Id associated to each object class.
const std::string pd_obj_name = "035_power_drill";
const std::string mb_obj_name = "006_mustard_bottle";
const std::string sb_obj_name = "004_sugar_box";

int main(int argc, char* argv[]) { 
    // Get the directories paths of the models.
    std::string scene_image_path{};  // Scene image path.
    std::string label_scene_path{};  // Label associated to the scene.
    std::string pd_models_dirpath{}; // Power drill models dir path
    std::string mb_models_dirpath{}; // Mustar bottle models dir path
    std::string sb_models_dirpath{}; // Sugar box models dir path
    parse_command_line(argc, argv, pd_models_dirpath, mb_models_dirpath, sb_models_dirpath, scene_image_path, label_scene_path);

    if (pd_models_dirpath.empty() || mb_models_dirpath.empty() 
            || sb_models_dirpath.empty() || scene_image_path.empty()
            || label_scene_path.empty()) {
        std::cerr << "Error in parsing the command line... aborting.\n";
        return 1;
    }

    // Get the path of each models inside the speficied directories.
    std::map<std::string, std::vector<std::string>> images_models_paths;
    get_all_filenames(pd_models_dirpath, images_models_paths[pd_obj_name]);
    get_all_filenames(mb_models_dirpath, images_models_paths[mb_obj_name]);
    get_all_filenames(sb_models_dirpath, images_models_paths[sb_obj_name]);

    // Check if one of the vector containing the paths of the models is empty.
    for (const auto& elem : images_models_paths) {
        if (elem.second.empty()) {
            std::cerr << "Error: one of the models directory is empty!" << std::endl;
            return -1;
        }
    }

    // Define the box color associated to each object.
    cv::Scalar color = cv::Scalar(0, 255, 0);
    std::map<std::string, cv::Scalar> boxes_color;
    boxes_color[pd_obj_name] = cv::Scalar(0, 0, 255);
    boxes_color[mb_obj_name] = cv::Scalar(255, 0, 0);
    boxes_color[sb_obj_name] = cv::Scalar(0, 255, 0);

    // Define the parameters associated to each object class.
    std::map<std::string, std::vector<float>> params_map;
    params_map[pd_obj_name] = pd_params;
    params_map[mb_obj_name] = mb_params;
    params_map[sb_obj_name] = sb_params;

    // Define the output scene image (the one with the boxes plotted).
    cv::Mat out_scene = cv::imread(scene_image_path, cv::IMREAD_COLOR);
    if(out_scene.empty()) {
        std::cerr << "Error: the image of the scene was not loaded correctly!" << std::endl;
        return -1;
    }
    cv::Mat out_scene_gray;
    cv::cvtColor(out_scene, out_scene_gray, cv::COLOR_BGR2GRAY);

    // Define the feature extractor that will be used to detect the objects.
    FeaturesExctractor extractor = FeaturesExctractor();

    // Define the vectors containing the keypoints and the descriptors of the scene.
    std::vector<cv::KeyPoint> keypoints_scene;
    cv::Mat descriptors_scene;
    extractor.extract_features(out_scene_gray, keypoints_scene, descriptors_scene);

    // Define the feature matcher.
    FeaturesMatcher matcher = FeaturesMatcher();

    // Define the models images.
    for(const auto& models_path : images_models_paths){
 
        std::cout << "Looking for " << models_path.first << " in the scene image..." << std::endl;
        // Define a vector containing all the models images of the current object.
        std::vector<cv::Mat> models;
        for (const std::string& p : models_path.second) {
            cv::Mat model = cv::imread(p, cv::IMREAD_GRAYSCALE);
            if(model.empty()){
                std::cerr<<"Error: the model image " << p << "was not loaded correctly!"<<std::endl;
                return -1;
            }
            models.push_back(model);
        }

        // Keypoints and descriptor of each model.
        std::vector<std::vector<cv::KeyPoint>> keypoints_models(models.size());
        std::vector<cv::Mat> descriptors_models(models.size());
        // Compute keypoints and descriptors.
        for (int i = 0; i < models.size(); i++) {
            extractor.extract_features(models[i], keypoints_models[i], descriptors_models[i]);
        }

        // Vector that stores matches found by the matcher.
        std::vector<std::vector<cv::DMatch>> knn_matches;
        // Compute the matches between the scene and each model.
        for (int i = 0; i < models.size(); i++) {
            std::vector<std::vector<cv::DMatch>> tmp;
            matcher.compute_matches(tmp, descriptors_models[i], descriptors_scene);
            knn_matches.insert(knn_matches.end(), tmp.begin(), tmp.end());
        }

        // Apply the first filter to the matches found previousliy.
        // The first filter is the lowe's filter.
        const float ratio_thresh = params_map[models_path.first][0]; // First parameter.
        std::vector<cv::DMatch> good_matches;
        lowe_filter(knn_matches, ratio_thresh, good_matches);

        // Check if anyone survived.
        if (good_matches.empty()) {
            std::cout << "No matches survived the Lowe's ratio filter..." << std::endl;
            continue;
        }

        // Now we want to work only on the matched points found in the scene image.
        // Therefore we convert the vector of DMatch into a vector of Point2i
        // corresponding to the maches positions of the scene image.
        std::vector<cv::Point2i> good_points;
        for(const auto& match : good_matches) {
            // Find the position of the match in the scene.
            cv::Point2i pt = keypoints_scene[match.trainIdx].pt;
            // Add the position only if it is not already present (aka remove duplicates).
            if(std::find(good_points.begin(), good_points.end(), pt) == good_points.end()) {
                good_points.push_back(pt);
            }
        }

        // Compute the center of mass of the points that survived the first filter.
        cv::Point2i com = compute_com(good_points);

        // Apply the second filter based on the position of the center of mass
        // (COM) of the remaining points.
        // All the points that have a distance from the COM that is bigger than
        // 'max_dist_from_com' are filtered out.
        float max_dist_from_com = params_map[models_path.first][1]; // Second parameter.
        std::vector<cv::Point2i> filtered_points;
        max_distance_filter(max_dist_from_com, good_points, com, filtered_points);

        // Check if anyone survived.
        if (filtered_points.empty()) {
            std::cout << "No matches survived the second filter (distance from COM)..." << std::endl;
            continue;
        }

        // Third filter: remove the isolated points.
        // Compute the number of neighbor considering 'max_dist_from_neighbor'
        // as max value. Then neglect the point if the number of 
        // neighbors is less then 'params_map[models_path.first][3]'.
        std::vector<cv::Point2i> final_points;
        int max_dist_from_neighbor = params_map[models_path.first][2]; // Third parameter.
        neighbor_filter(max_dist_from_neighbor, params_map[models_path.first][3], filtered_points, final_points);

        // Printing the dimensione of the matches.
        int num_points = final_points.size();
        std::cout << "Final points survived: " << num_points << std::endl;

        // Draw the filtered matches.
        //for (const auto& pt : final_points) {
        //    cv::circle(out_scene, pt, 3, boxes_color[models_path.first], 2);
        //}

        // Recalculate the center of mass with the filtered matches.
        cv::Point2i new_com = compute_com(final_points);

        // Draw the center of mass.
        //cv::circle(out_scene, new_com, 5, cv::Scalar(0, 255, 255), 2);

        // Get the value of top left corner and bottom right bottom of the box.
        const float expansion_ratio = 0.1; // Used to expand the box.
        std::pair<cv::Point2i, cv::Point2i> label = bounding_box_coord(out_scene, final_points, keypoints_scene, expansion_ratio);
        cv::Point2i top_left = label.first;
        cv::Point2i bottom_right = label.second;
        
        // Compute the are inside the box.
        double x_dim = label.first.x - label.second.x;
        double y_dim = label.first.y - label.second.y;
        double area = (x_dim * y_dim) / scale_factor; // Scale the area.

        // Draw the box, if necessary.
        if (area != 0) {
            // Compute the density.
            double density  = num_points / area;
            std::cout<<"Density value is " << density <<std::endl;

            // If density and number of points are high enough, then draw the box.
            if (density > params_map[models_path.first][4] && num_points >= params_map[models_path.first][5]) {
                rectangle(out_scene, top_left, bottom_right, boxes_color[models_path.first], 2, cv::LINE_8);  
                // Store the found label.
                store_label("output_label.txt", models_path.first, label.first, label.second);
            }
        }

    }

    // Compute the metrics.
    PerformanceMetrics metrics = PerformanceMetrics("output_label.txt", label_scene_path);
    std::cout<< std::endl;
    metrics.print_metrics();
    std::cout<< std::endl;

    // Plot the final result.
    cv::imshow("Filtered Matches on Scene", out_scene);
    cv::waitKey();
    return 0;
}
