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
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using std::cout;
using std::endl;

int main(int argc, char* argv[]) {

    // Models images.
    std::string models_paths[] = {
        "../data/035_power_drill/models/view_0_001_color.png",
        "../data/035_power_drill/models/view_0_002_color.png",
        "../data/035_power_drill/models/view_0_003_color.png",
        "../data/035_power_drill/models/view_0_004_color.png",
        "../data/035_power_drill/models/view_0_005_color.png",
        "../data/035_power_drill/models/view_0_006_color.png",
        "../data/035_power_drill/models/view_0_007_color.png",
        "../data/035_power_drill/models/view_0_008_color.png",
        "../data/035_power_drill/models/view_0_009_color.png",
        "../data/035_power_drill/models/view_30_000_color.png",
        "../data/035_power_drill/models/view_30_001_color.png",
        "../data/035_power_drill/models/view_30_002_color.png",
        "../data/035_power_drill/models/view_30_003_color.png",
        "../data/035_power_drill/models/view_30_004_color.png",
        "../data/035_power_drill/models/view_30_005_color.png",
        "../data/035_power_drill/models/view_30_006_color.png",
        "../data/035_power_drill/models/view_30_007_color.png",
        "../data/035_power_drill/models/view_30_008_color.png",
        "../data/035_power_drill/models/view_30_009_color.png",
        "../data/035_power_drill/models/view_60_000_color.png",
        "../data/035_power_drill/models/view_60_001_color.png",
        "../data/035_power_drill/models/view_60_002_color.png",
        "../data/035_power_drill/models/view_60_003_color.png",
        "../data/035_power_drill/models/view_60_004_color.png",
        "../data/035_power_drill/models/view_60_005_color.png",
        "../data/035_power_drill/models/view_60_006_color.png",
        "../data/035_power_drill/models/view_60_007_color.png",
        "../data/035_power_drill/models/view_60_008_color.png",
        "../data/035_power_drill/models/view_60_009_color.png",
    };

    // Scene image path.
    std::string scene_path = "../data/035_power_drill/test_images/35_0077_000519-color.jpg";

    // Define the models images.
    std::vector<Mat> models;
    for (const auto& p : models_paths) {
        models.push_back(imread(p, IMREAD_GRAYSCALE));
    }

    // Define the scene image.
    Mat scene = imread(scene_path, IMREAD_GRAYSCALE);

    blur(scene, scene, Size(3, 3));

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
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);

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
    const float ratio_thresh = 0.75;
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
    float max_distance_threshold = 170; // Adjust this threshold as needed
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
    //Mat scene_with_centers;
    //cvtColor(scene, scene_with_centers, COLOR_GRAY2BGR);
    Mat scene_matches;
    cvtColor(scene, scene_matches, COLOR_GRAY2BGR);
    Scalar color_center = Scalar(0, 0, 255);
    int radius_center = 5;
    int thickness_center = 2;
    if (total_num_points > 0) {
        circle(scene_matches, new_com, radius_center, color_center, thickness_center);
    }


    // -- Draw the filtered matches
    //Mat scene_matches;
    //cvtColor(scene, scene_matches, COLOR_GRAY2BGR);
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

    imshow("Filtered Matches on Scene", scene_matches);

    waitKey();
    return 0;
}
//004_sugar_box 366 172 504 446
