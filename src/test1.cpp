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
#include "../include/utils.h"
#include "../include/features_extractor.h"
#include "../include/features_matcher.h"
#include "../include/performance_metrics.h"

using namespace cv;
using std::cout;
using std::endl;

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

    // Define the models images.
    for(const auto& models_paths : std::vector<std::vector<std::string>>{pd_models_images_paths, 
        mb_models_images_paths, sb_models_images_paths}){

        std::vector<Mat> models;
        for (const auto& p : models_paths) {
            Mat model =  imread(p, IMREAD_GRAYSCALE);
            if(model.empty()){
            std::cerr<<"Error: the image of the model was not loaded correctly!"<<std::endl;
            return -1;
            }
            models.push_back(imread(p, IMREAD_GRAYSCALE));
            
        }
        

        // Define the scene image.
        Mat scene = imread(scene_image_path, IMREAD_GRAYSCALE);
        if(scene.empty()){
        std::cerr<<"Error: the image of the scene was not loaded correctly!"<<std::endl;
        return -1;
        }



        //-- STEP 1: Detect the keypoints using detector/extractor, compute the descriptors
        //           and keypoints.
        // Define the detector/extractor.
        FeaturesExctractor extractor = FeaturesExctractor();

        // Keypoints and descriptor of each model.
        std::vector<std::vector<KeyPoint>> keypoints_models(models.size());
        std::vector<Mat> descriptors_models(models.size());

        // Keypoints and descriptor of the scene.
        std::vector<KeyPoint> keypoints_scene;
        Mat descriptors_scene;

        // Compute keypoints and descriptors using detector/extractor.
        for (int i = 0; i < models.size(); i++) {
        extractor.extract_features(models[i], keypoints_models[i], descriptors_models[i]);
        }
        extractor.extract_features(scene, keypoints_scene, descriptors_scene);



        //-- STEP 2: Matching descriptor vectors with a BRUTEFORCE matcher.
        // Define the matcher.
        FeaturesMatcher matcher = FeaturesMatcher();

        // Vector that stores the good matches.
        std::vector<std::vector<DMatch>> knn_matches;

        // Compute the matches between the scene and each model.
        for (int i = 0; i < models.size(); i++) {
            std::vector<std::vector<DMatch>> tmp;
            matcher.compute_matches(tmp, descriptors_models[i],descriptors_scene);
            knn_matches.insert(knn_matches.end(), tmp.begin(), tmp.end());
        }

        // Filter matches using the Lowe's ratio test.
        const float ratio_thresh = 0.7;
        std::vector<DMatch> good_matches;
        lowe_filter(knn_matches, ratio_thresh, good_matches);




        //-- STEP 3: Filter matches based on distance to the center of mass.
        // Calculate the total center of mass from all good matches.
        Point2f com = compute_com(good_matches, keypoints_scene);

        // Adjust this threshold as needed.
        float max_distance_threshold = 100; 
        std::vector<DMatch> filtered_matches;

        max_distance_filter(max_distance_threshold, good_matches, keypoints_scene, com, filtered_matches);

        // Recalculate the center of mass with the filtered matches.
        Point2f new_com = compute_com(filtered_matches, keypoints_scene);
        
        // If the new center of mass after computing the max_distance_filter is the fake one
        // then the new com remains equal to the old one.
        if(new_com.x == fake_com.x || new_com.y == fake_com.y){
        new_com = com;
        }


        // STEP 4: Visualization of the results.
        // Visualize the center of mass.
        Mat scene_with_centers;
        cvtColor(scene, scene_with_centers, COLOR_GRAY2BGR);
        Scalar color_center = Scalar(0, 0, 255);
        int radius_center = 5;
        int thickness_center = 2;
        int total_num_points = filtered_matches.size();
        if (total_num_points > 0)
            circle(scene_with_centers, new_com, radius_center, color_center, thickness_center);
        imshow("Scene with Filtered Center of Mass", scene_with_centers);


        // Draw the filtered matches.
        Mat scene_matches;
        cvtColor(scene, scene_matches, COLOR_GRAY2BGR);
        Scalar color = Scalar(0, 255, 0);
        int radius = 3;
        int thickness = 2;
        for (const auto& match : filtered_matches) {
            Point2f pt2 = keypoints_scene[match.trainIdx].pt;
            circle(scene_matches, pt2, radius, color, thickness);
        }

        // Last step draw the rectangle around all the matches.
        draw_box(scene_matches, filtered_matches, keypoints_scene);
        imshow("Filtered Matches on Scene", scene_matches);

        waitKey();
    }
    return 0;
}
