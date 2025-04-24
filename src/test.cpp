#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include <set> 
#include <vector>
#include <map> 


// FUNCTION DEFINITIONS: 

// Function to perform Mean Shift segmentation
cv::Mat performMeanShiftSegmentation(const cv::Mat& input) {
    
    cv::Mat shifted;
    
    // Parameters for Mean Shift
    int spatialWindowRadius = 30;
    int colorWindowRadius = 35;
    
    // Apply Mean Shift
    pyrMeanShiftFiltering(input, shifted, spatialWindowRadius, colorWindowRadius, 2);
    
    return shifted;
}

// --- Vec3bCompare struct (ensure it's defined) ---
struct Vec3bCompare {
    bool operator()(const cv::Vec3b& a, const cv::Vec3b& b) const {
        if (a[0] != b[0]) return a[0] < b[0];
        if (a[1] != b[1]) return a[1] < b[1];
        return a[2] < b[2];
    }
};

std::vector<cv::Mat> findSegmentsWithMatches(const cv::Mat& segmented_bgr,
                                        const std::vector<cv::DMatch>& matches,
                                        const std::vector<cv::KeyPoint>& sceneKeypoints)
{
    std::vector<cv::Mat> activatedSegmentMasks;

    // --- Input Validation ---
    if (segmented_bgr.empty() || segmented_bgr.type() != CV_8UC3) {
        std::cerr << "Error: Segmented image is invalid (empty or not CV_8UC3)." << std::endl;
        return activatedSegmentMasks;
    }
    if (matches.empty()) {
         std::cerr << "Warning: No matches provided to findSegmentsWithMatches." << std::endl;
         return activatedSegmentMasks;
    }
    if (sceneKeypoints.empty()){
        std::cerr << "Warning: Scene keypoints are empty." << std::endl;
        return activatedSegmentMasks;
    }

    // --- 1. Identify unique segment colors containing matches ---
    std::set<cv::Vec3b, Vec3bCompare> colors_with_matches;

    for (const auto& match : matches) {
        // Validate index from match object
        if (match.trainIdx < 0 || match.trainIdx >= sceneKeypoints.size()) {
            std::cerr << "Warning: Invalid match trainIdx: " << match.trainIdx << std::endl;
            continue;
        }

        // Get the integer coordinates of the keypoint for pixel access
        cv::Point pt = sceneKeypoints[match.trainIdx].pt;

        // Check if point is within image bounds
        if (pt.x >= 0 && pt.y >= 0 && pt.x < segmented_bgr.cols && pt.y < segmented_bgr.rows) {
            // Get the color (segment identifier) from the Mean Shift output image
            cv::Vec3b segment_color = segmented_bgr.at<cv::Vec3b>(pt);
            // Add the color to the set. std::set automatically handles uniqueness.
            colors_with_matches.insert(segment_color);
        } else {
             std::cerr << "Warning: Match keypoint (" << pt.x << ", " << pt.y
                       << ") is outside segmented image bounds (" << segmented_bgr.cols << "x" << segmented_bgr.rows << ")." << std::endl;
        }
    }

    if (colors_with_matches.empty()) {
        std::cout << "No matches found within the bounds of any segment." << std::endl;
        return activatedSegmentMasks;
    }

    std::cout << "Found " << colors_with_matches.size() << " unique segment colors containing matches." << std::endl;

    // --- 2. Create a mask for each unique activated segment color ---
    for (const auto& color : colors_with_matches) {
        cv::Mat segment_mask;
        // Create a mask selecting all pixels in the segmented image with this exact color
        cv::inRange(segmented_bgr, color, color, segment_mask);

        // Optional check: Ensure the mask is not empty (shouldn't happen if color came from a pixel)
        if (cv::countNonZero(segment_mask) == 0) {
            std::cerr << "Warning: Mask for color " << color << " became empty unexpectedly." << std::endl;
            continue;
        }

        // --- Optional: Clean up the mask ---
        // You might still want to fill small holes or remove tiny noise specs
        // within the activated segment.
        cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15,15)); // Adjust kernel size as needed
        morphologyEx(segment_mask, segment_mask, cv::MORPH_CLOSE, kernel);
        medianBlur(segment_mask,segment_mask, 5);
        // morphologyEx(segment_mask, segment_mask, MORPH_OPEN, kernel); // Be cautious with OPEN

         if (cv::countNonZero(segment_mask) > 0) { // Check again after morphology
            activatedSegmentMasks.push_back(segment_mask);
         } else {
             std::cerr << "Warning: Mask for color " << color << " became empty after morphology." << std::endl;
         }
    }

    std::cout << "Returning " << activatedSegmentMasks.size() << " activated segment masks." << std::endl;
    return activatedSegmentMasks;
}


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
    //std::string scene_path = "../data/035_power_drill/test_images/4_0058_000001-color.jpg";
    //std::string scene_path = "../data/035_power_drill/test_images/4_0077_000659-color.jpg";
    //std::string scene_path = "../data/035_power_drill/test_images/35_0038_002606-color.jpg";
    //std::string scene_path = "../data/035_power_drill/test_images/35_0010_001853-color.jpg";
    //std::string scene_path = "../data/035_power_drill/test_images/35_0030_001009-color.jpg";
    std::string scene_path = "../data/035_power_drill/test_images/35_0030_000046-color.jpg";
   
    // Define the models images.
    std::vector<cv::Mat> models;
    for (const auto& p : models_paths) {
        models.push_back(cv::imread(p, cv::IMREAD_GRAYSCALE));
    }

    // Define the scene image.
    cv::Mat scene = imread(scene_path, cv::IMREAD_GRAYSCALE);

    //-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors.

    // Define the SIFT detector.
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();

    // Keypoints and descriptor of each model.
    std::vector<std::vector<cv::KeyPoint>> keypoints_models(models.size());
    std::vector<cv::Mat> descriptors_models(models.size());

    // Keypoints and descriptor of the scene.
    std::vector<cv::KeyPoint> keypoints_scene;
    cv::Mat descriptors_scene;

    // Compute keypoints and descriptors.
    for (int i = 0; i < models.size(); i++) {
        detector->detectAndCompute(models[i], cv::noArray(), keypoints_models[i], descriptors_models[i]);
    }
    detector->detectAndCompute(scene, cv::noArray(), keypoints_scene, descriptors_scene);

    //-- Step 2: Matching descriptor vectors with a BRUTEFORCE matcher

    // Define the Brute force matcher.
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);

    // Vector that stores the good matches.
    //std::vector<std::vector< std::vector<DMatch> >> knn_matches(models.size());
    std::vector<std::vector<cv::DMatch>> knn_matches;

    // Compute the matches between the scene and each model.
    for (int i = 0; i < models.size(); i++) {
        std::vector<std::vector<cv::DMatch>> tmp;
        matcher->knnMatch(descriptors_models[i], descriptors_scene, tmp, 2);
        knn_matches.insert(knn_matches.end(), tmp.begin(), tmp.end());
    }

    // Filter matches using the Lowe's ratio test.
    const float ratio_thresh = 0.8;
    std::vector<cv::DMatch> good_matches;
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
        cv::Point2f pt = keypoints_scene[match.trainIdx].pt;
        total_sum_x += pt.x;
        total_sum_y += pt.y;
        total_num_points++;
    }

    cv::Point2f com;
    if (total_num_points > 0) {
        com.x = total_sum_x / total_num_points;
        com.y = total_sum_y / total_num_points;
        std::cout << "Total Center of Mass: (" << com.x << ", " << com.y << ")" << std::endl;
    }
    else {
        std::cout << "No good matches found at all." << std::endl;
        com.x = -1;
        com.y = -1;
    }

    //-- Step 3: Filter matches based on distance to the center of mass
    float max_distance_threshold = 100; // Adjust this threshold as needed
    std::vector<cv::DMatch> filtered_matches;;

    total_sum_x = 0;
    total_sum_y = 0;
    total_num_points = 0;
    for (const auto& match : good_matches) {
        cv::Point2f pt = keypoints_scene[match.trainIdx].pt;
        float distance = sqrt(pow(pt.x - com.x, 2) + pow(pt.y - com.y, 2));
        if (distance <= max_distance_threshold) {
            filtered_matches.push_back(match);
            total_sum_x += pt.x;
            total_sum_y += pt.y;
            total_num_points++;
        }
    }

    // Recalculate the center of mass with the filtered matches
    cv::Point2f new_com;
    if (total_num_points > 0) {
        new_com.x = total_sum_x / total_num_points;
        new_com.y = total_sum_y / total_num_points;
        std::cout << "Filtered Center of Mass: (" << new_com.x << ", " << new_com.y << ")" << std::endl;
    }
    else {
        std::cout << "No matches survived the filtering." << std::endl;
        new_com = com; // Keep the old COM or set to a default value
    }

    // Visualize the center of mass.
    cv::Mat scene_with_centers;
    cvtColor(scene, scene_with_centers, cv::COLOR_GRAY2BGR);
    cv::Scalar color_center = cv::Scalar(0, 0, 255);
    int radius_center = 5;
    int thickness_center = 2;
    if (total_num_points > 0)
        cv::circle(scene_with_centers, new_com, radius_center, color_center, thickness_center);
        cv::imshow("Scene with Filtered Center of Mass", scene_with_centers);


    // -- Draw the filtered matches
    cv::Mat scene_matches;
    cv::cvtColor(scene, scene_matches, cv::COLOR_GRAY2BGR);
    cv::Scalar color = cv::Scalar(0, 255, 0);
    int radius = 3;
    int thickness = 2;
    for (const auto& match : filtered_matches) {
        cv::Point2f pt2 = keypoints_scene[match.trainIdx].pt;
        cv::circle(scene_matches, pt2, radius, color, thickness);
    }

    // Last step draw the rectangle.
    cv::Point2f topLeft(scene.rows, scene.cols);
    cv::Point2f bottomRight(0, 0);
    for (const auto& match : filtered_matches) {
        cv::Point2f pt = keypoints_scene[match.trainIdx].pt;
        if (pt.y < topLeft.y)
            topLeft.y = pt.y;
        if (pt.x < topLeft.x)
            topLeft.x = pt.x;
        if (pt.y > bottomRight.y)
            bottomRight.y = pt.y;
        if (pt.x > bottomRight.x)
            bottomRight.x = pt.x;
    }

    cv::rectangle(scene_matches, topLeft, bottomRight, cv::Scalar(255, 0, 0), 2, cv::LINE_8); 
    // After computing filtered_matches and new_com:
    if (total_num_points > 0) { // Check if filtered matches exist before drawing CoM
        cv::circle(scene_with_centers, new_com, radius_center, color_center, thickness_center);
    }
    // imshow("Scene with Filtered Center of Mass", scene_with_centers); // Can show this intermediate step if needed


    // Load the scene image in color for segmentation
    cv::Mat sceneColor = cv::imread(scene_path, cv::IMREAD_COLOR);
     if(sceneColor.empty()){
        std::cout << "Error: Could not load color scene image: " << scene_path << std::endl;
        return -1;
    }

    // Perform Mean Shift segmentation
    std::cout << "Performing Mean Shift..." << std::endl;
    cv::Mat segmented = performMeanShiftSegmentation(sceneColor); // Uses hardcoded params inside function now
    cv::imshow("Mean Shift Segmentation Output", segmented); // Show raw segmentation

    // Find masks for all segments containing at least one filtered match
    std::cout << "Finding segments containing matches..." << std::endl;
    std::vector<cv::Mat> segmentMasks = findSegmentsWithMatches(segmented, filtered_matches,
        keypoints_scene);

    // --- Process the combined segments ---
    cv::Mat paintedSegments; // Image to show the painted segments
    cv::Rect overallBoundingBox; // Single bounding box for all segments

    if (!segmentMasks.empty()) {
        // 1. Combine all individual masks into one <<<<<<<<<<<<< COMBINING MASKS
        cv::Mat combinedMask = cv::Mat::zeros(sceneColor.size(), CV_8U);
        for (const auto& mask : segmentMasks) {
            if (!mask.empty() && mask.size() == combinedMask.size()) { // Basic check
                cv::bitwise_or(combinedMask, mask, combinedMask);
            }
        }
        cv::imshow("Combined Segment Mask", combinedMask); // Visualize the combined mask

        // 2. Paint the segments onto a result image using the combined mask <<<<<< PAINTING COMBINED
        paintedSegments = cv::Mat::zeros(sceneColor.size(), sceneColor.type());
        sceneColor.copyTo(paintedSegments, combinedMask); // Use COMBINED mask

        // 3. Find contours on the combined mask to get the overall bounding box <<<<<<< OVERALL BOX
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(combinedMask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            // Calculate a single bounding box encompassing all found contours
            overallBoundingBox = cv::boundingRect(contours[0]);
            for (size_t i = 1; i < contours.size(); ++i) {
                overallBoundingBox |= cv::boundingRect(contours[i]); // Merge Rects
            }

            // 4. Draw the single overall bounding box <<<<<<<<<<< DRAWING SINGLE BOX
            cv::Scalar bbox_color(0, 255, 0); // Green for the overall box
            int bbox_thickness = 2;
            cv::rectangle(paintedSegments, overallBoundingBox, bbox_color, bbox_thickness); // Draw on painted image
            cv::rectangle(scene_with_centers, overallBoundingBox, bbox_color, bbox_thickness); // Draw on CoM image

            // Optional: Count total matches within the final combined area
            int totalMatchesInCombined = 0;
            // ... (code to count matches in combinedMask) ...
            //putText(scene_with_centers, std::to_string(totalMatchesInCombined) + " matches in segments",
            //      overallBoundingBox.tl() + Point(0, -5), FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 1); // SINGLE text label

        } else {
            std::cout << "Warning: No contours found on the combined segment mask." << std::endl;
        }

    } else {
        std::cout << "No segments containing matches were found." << std::endl;
        paintedSegments = cv::Mat::zeros(sceneColor.size(), sceneColor.type());
    }

    // --- Final Display ---
    // NO loop showing individual segments here.

    cv::imshow("Painted Activated Segments", paintedSegments); // Shows the painted result + overall box
    cv::imshow("Final Detection (CoM + Overall BBox)", scene_with_centers); // Shows CoM + overall box on grayscale

    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}