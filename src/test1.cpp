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


int main(int argc, char* argv[]) { 
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

    // Get the path of each models inside the speficied directories.
    std::map<std::string, std::vector<std::string>> images_models_paths;
    get_all_filenames(pd_models_dirpath, images_models_paths["powerdrill"]);
    get_all_filenames(mb_models_dirpath, images_models_paths["mustardbottle"]);
    get_all_filenames(sb_models_dirpath, images_models_paths["sugarbox"]);

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
    boxes_color["powerdrill"] = cv::Scalar(255, 0, 0);
    boxes_color["mustardbottle"] = cv::Scalar(0, 255, 0);
    boxes_color["sugarbox"] = cv::Scalar(0, 0, 255);

    // Define the output scene image (the one with the boxes plotted).
    cv::Mat out_scene = cv::imread(scene_image_path, cv::IMREAD_GRAYSCALE);
    if(out_scene.empty()) {
        std::cerr << "Error: the image of the scene was not loaded correctly!" << std::endl;
        return -1;
    }
    cv::cvtColor(out_scene, out_scene, cv::COLOR_GRAY2BGR);

    // Define the feature extractor that will be used to detect the objects.
    FeaturesExctractor extractor = FeaturesExctractor();

    // Define the vectors containing the keypoints and the descriptors of the scene.
    std::vector<cv::KeyPoint> keypoints_scene;
    cv::Mat descriptors_scene;
    extractor.extract_features(out_scene, keypoints_scene, descriptors_scene);

    // Define the feature matcher.
    FeaturesMatcher matcher = FeaturesMatcher();

    // Define the models images.
    for(const auto& models_path : images_models_paths){
        std::cout << "Looking for " << models_path.first << " in the scene image..." << std::endl;
        // Define a vector containing all the models images of the current object.
        std::vector<cv::Mat> models;
        for (const std::string& p : models_path.second) {
            cv::Mat model =  cv::imread(p, cv::IMREAD_GRAYSCALE);
            if(model.empty()){
                std::cerr<<"Error: the image of the model was not loaded correctly!"<<std::endl;
                return -1;
            }
            models.push_back(cv::imread(p, cv::IMREAD_GRAYSCALE));
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

        // Filter matches using the Lowe's ratio test.
        const float ratio_thresh = 0.7;
        std::vector<cv::DMatch> good_matches;
        lowe_filter(knn_matches, ratio_thresh, good_matches);



        //-- STEP 3: Filter matches based on distance to the center of mass.
        // Calculate the total center of mass from all good matches.
        cv::Point2f com = compute_com(good_matches, keypoints_scene);

        // Adjust this threshold as needed.
        float max_distance_threshold = 100; 
        std::vector<cv::DMatch> filtered_matches;
        max_distance_filter(max_distance_threshold, good_matches, keypoints_scene, com, filtered_matches);

        if (filtered_matches.empty()) {
            std::cout << "No matches survived the max distance filter..." << std::endl;
            continue;
        }

        // Draw the filtered matches.
        for (const auto& match : filtered_matches) {
            cv::Point2f pt2 = keypoints_scene[match.trainIdx].pt;
            cv::circle(out_scene, pt2, 3, boxes_color[models_path.first], 2);
        }

        // Recalculate the center of mass with the filtered matches.
        cv::Point2f new_com = compute_com(filtered_matches, keypoints_scene);
        // Draw the center of mass.
        cv::circle(out_scene, new_com, 5, cv::Scalar(0, 255, 255), 2);

        // Last step draw the rectangle around all the matches.
        draw_box(out_scene, filtered_matches, keypoints_scene, boxes_color[models_path.first]);
    }

    // Plot the final result.
    cv::imshow("Filtered Matches on Scene", out_scene);
    cv::waitKey();
    return 0;
}
