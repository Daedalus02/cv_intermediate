/*
 * Approach used:
 * 1) Compute feature matching between all the 30 rgb models and the scene image.
 * 2) Filter the matches found in step (1) with Lowes ratio test.
 * 3) Compute the center of mass (COM) of the matches located in the scene image (green circles).
 * 4) Neglect all the matches that are far away from the COM.
 * 5) Update the position of the COM of the matches, then draw it in red.
 */

#include <iostream>
#include <vector>
#include <string>
#include <cmath> // For sqrt, pow

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

// Define a struct to pass data to the trackbar callback
struct CallbackData {
    Mat scene_img;
    std::vector<KeyPoint> keypoints_scene;
    std::vector<std::vector<DMatch>> knn_matches;
    float max_distance_threshold = 120; // Initial value for distance threshold
    // We will draw on a copy of the original scene image
    Mat scene_display;
};

// Global variable for the ratio threshold trackbar position
int ratio_threshold_slider = 78; // Initial value (corresponds to 0.75)
const int ratio_threshold_slider_max = 100;

// Global variable for the max distance threshold trackbar position
int max_distance_slider = 120; // Initial value
const int max_distance_slider_max = 1000;
const int max_distance_slider_min = 1;

// Callback function for the ratio threshold trackbar
void on_ratio_threshold_trackbar(int, void* userdata) {
    CallbackData* data = static_cast<CallbackData*>(userdata);

    // Get the current ratio threshold from the trackbar value
    float ratio_thresh = ratio_threshold_slider / 100.0f;

    // Filter matches using the updated Lowe's ratio test.
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < data->knn_matches.size(); i++) {
        if (data->knn_matches[i].size() > 1 && data->knn_matches[i][0].distance < ratio_thresh * data->knn_matches[i][1].distance) {
            good_matches.push_back(data->knn_matches[i][0]);
        }
    }

    // Calculate the total center of mass from all good matches
    double total_sum_x = 0;
    double total_sum_y = 0;
    int total_num_points = 0;
    for (const auto& match : good_matches) {
        Point2f pt = data->keypoints_scene[match.trainIdx].pt;
        total_sum_x += pt.x;
        total_sum_y += pt.y;
        total_num_points++;
    }

    Point2f com;
    if (total_num_points > 0) {
        com.x = total_sum_x / total_num_points;
        com.y = total_sum_y / total_num_points;
        //cout << "Total Center of Mass: (" << com.x << ", " << com.y << ")" << endl;
    } else {
        //cout << "No good matches found at all." << endl;
        com.x = -1;
        com.y = -1;
    }

    //-- Step 3: Filter matches based on distance to the center of mass
    // Note: This step depends on the initial COM. A more robust approach might re-calculate COM iteratively.
    // For simplicity, we keep the original logic of filtering based on the COM of the current good_matches.
    std::vector<DMatch> filtered_matches;

    total_sum_x = 0;
    total_sum_y = 0;
    total_num_points = 0;
    if (com.x != -1) { // Only filter if a COM was calculated
        for (const auto& match : good_matches) {
            Point2f pt = data->keypoints_scene[match.trainIdx].pt;
            float distance = std::sqrt(std::pow(pt.x - com.x, 2) + std::pow(pt.y - com.y, 2));
            if (distance <= data->max_distance_threshold) {
                filtered_matches.push_back(match);
                total_sum_x += pt.x;
                total_sum_y += pt.y;
                total_num_points++;
            }
        }
    }


    // Recalculate the center of mass with the filtered matches
    Point2f new_com;
    if (total_num_points > 0) {
        new_com.x = total_sum_x / total_num_points;
        new_com.y = total_sum_y / total_num_points;
        //cout << "Filtered Center of Mass: (" << new_com.x << ", " << new_com.y << ")" << endl;
    } else {
        //cout << "No matches survived the filtering based on COM." << endl;
        new_com = com; // Keep the old COM or handle as needed
    }

    // Clear the previous drawing and draw on a fresh copy of the scene image
    cvtColor(data->scene_img, data->scene_display, COLOR_GRAY2BGR);

    // Visualize the center of mass.
    Scalar color_center = Scalar(0, 0, 255); // Red
    int radius_center = 5;
    int thickness_center = 2;
    if (total_num_points > 0)
        circle(data->scene_display, new_com, radius_center, color_center, thickness_center);


    // -- Draw the filtered matches (green circles)
    Scalar color_matches = Scalar(0, 255, 0); // Green
    int radius_match = 3;
    int thickness_match = 2;
    for (const auto& match : filtered_matches) {
        Point2f pt2 = data->keypoints_scene[match.trainIdx].pt;
        circle(data->scene_display, pt2, radius_match, color_matches, thickness_match);
    }

    // Last step draw the rectangle (blue).
    if (filtered_matches.size() > 0) {
        Point2f topLeft(data->scene_img.cols, data->scene_img.rows);
        Point2f bottomRight(0, 0);
        for (const auto& match : filtered_matches) {
            Point2f pt = data->keypoints_scene[match.trainIdx].pt;
            if (pt.y < topLeft.y)
                topLeft.y = pt.y;
            if (pt.x < topLeft.x)
                topLeft.x = pt.x;
            if (pt.y > bottomRight.y)
                bottomRight.y = pt.y;
            if (pt.x > bottomRight.x)
                bottomRight.x = pt.x;
        }
        rectangle(data->scene_display, topLeft, bottomRight, Scalar(255, 0, 0), 2, LINE_8);
    }


    // Display the updated image
    imshow("Object Detection", data->scene_display);
}

// Callback function for the max distance threshold trackbar
void on_max_distance_threshold_trackbar(int pos, void* userdata) {
    CallbackData* data = static_cast<CallbackData*>(userdata);
    data->max_distance_threshold = static_cast<float>(pos);
    on_ratio_threshold_trackbar(ratio_threshold_slider, data); // Redraw with the new distance threshold
}


int main(int argc, char* argv[]) {

    // Models images.
    std::string models_paths[] = {
        "../data/004_sugar_box//models/view_0_001_color.png",
        "../data/004_sugar_box//models/view_0_002_color.png",
        "../data/004_sugar_box//models/view_0_003_color.png",
        "../data/004_sugar_box//models/view_0_004_color.png",
        "../data/004_sugar_box//models/view_0_005_color.png",
        "../data/004_sugar_box//models/view_0_006_color.png",
        "../data/004_sugar_box//models/view_0_007_color.png",
        "../data/004_sugar_box//models/view_0_008_color.png",
        "../data/004_sugar_box//models/view_0_009_color.png",
        "../data/004_sugar_box//models/view_30_000_color.png",
        "../data/004_sugar_box//models/view_30_001_color.png",
        "../data/004_sugar_box//models/view_30_002_color.png",
        "../data/004_sugar_box//models/view_30_003_color.png",
        "../data/004_sugar_box//models/view_30_004_color.png",
        "../data/004_sugar_box//models/view_30_005_color.png",
        "../data/004_sugar_box//models/view_30_006_color.png",
        "../data/004_sugar_box//models/view_30_007_color.png",
        "../data/004_sugar_box//models/view_30_008_color.png",
        "../data/004_sugar_box//models/view_30_009_color.png",
        "../data/004_sugar_box//models/view_60_000_color.png",
        "../data/004_sugar_box//models/view_60_001_color.png",
        "../data/004_sugar_box//models/view_60_002_color.png",
        "../data/004_sugar_box//models/view_60_003_color.png",
        "../data/004_sugar_box//models/view_60_004_color.png",
        "../data/004_sugar_box//models/view_60_005_color.png",
        "../data/004_sugar_box//models/view_60_006_color.png",
        "../data/004_sugar_box//models/view_60_007_color.png",
        "../data/004_sugar_box//models/view_60_008_color.png",
        "../data/004_sugar_box//models/view_60_009_color.png",
    };

    // Scene image path.
    std::string scene_path = "../data/004_sugar_box/test_images/4_0058_000001-color.jpg";

    // Define the models images.
    std::vector<Mat> models;
    for (const auto& p : models_paths) {
        models.push_back(imread(p, IMREAD_GRAYSCALE));
        if (models.back().empty()) {
            cout << "Could not read the model image: " << p << endl;
            return -1;
        }
    }

    // Define the scene image.
    Mat scene = imread(scene_path, IMREAD_GRAYSCALE);
    if (scene.empty()) {
        cout << "Could not read the scene image: " << scene_path << endl;
        return -1;
    }

    //blur(scene, scene, Size(3, 3));

    //-- Step 1: Detect the keypoints using BRISK Detector, compute the descriptors.

    // Define the BRISK detector.
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

    // Define the Brute force matcher (using Hamming distance for BRISK).
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);

    // Vector that stores the knn matches (k=2 for ratio test).
    std::vector<std::vector<DMatch>> knn_matches;

    // Compute the matches between each model and the scene.
    // We iterate through models and match their descriptors against the scene's descriptors.
    for (int i = 0; i < models.size(); i++) {
        // Ensure descriptors are not empty and have the same type if needed (though BRISK is binary)
        if (!descriptors_models[i].empty() && !descriptors_scene.empty()) {
            std::vector<std::vector<DMatch>> tmp_matches;
            matcher->knnMatch(descriptors_models[i], descriptors_scene, tmp_matches, 2);
            // Append the matches from the current model to the overall knn_matches vector
            knn_matches.insert(knn_matches.end(), tmp_matches.begin(), tmp_matches.end());
        } else {
            cout << "Warning: Descriptors are empty for model " << i << " or scene." << endl;
        }
    }


    // Prepare data for the callback
    CallbackData data;
    data.scene_img = scene; // Store original grayscale scene
    data.keypoints_scene = keypoints_scene;
    data.knn_matches = knn_matches;
    // Create a color copy for drawing
    cvtColor(scene, data.scene_display, COLOR_GRAY2BGR);


    // Create a window and trackbar
    const std::string window_name = "Object Detection";
    namedWindow(window_name);

    createTrackbar("Ratio Threshold (%)", window_name, &ratio_threshold_slider, ratio_threshold_slider_max, on_ratio_threshold_trackbar, &data);
    createTrackbar("Max Distance", window_name, &max_distance_slider, max_distance_slider_max, on_max_distance_threshold_trackbar, &data);
    setTrackbarMin("Max Distance", window_name, max_distance_slider_min);

    // Initial call to the trackbar callback to display the image with the default threshold
    on_ratio_threshold_trackbar(ratio_threshold_slider, &data);

    // Wait for a key press
    waitKey();

    return 0;
}
