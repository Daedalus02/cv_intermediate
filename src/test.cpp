#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;


int main( int argc, char* argv[] ) {

    std::string models_paths[] = {
        "../data/004_sugar_box/models/view_0_001_color.png",
        "../data/004_sugar_box/models/view_0_002_color.png",
        "../data/004_sugar_box/models/view_0_003_color.png",
        "../data/004_sugar_box/models/view_0_004_color.png",
        "../data/004_sugar_box/models/view_0_005_color.png",
        "../data/004_sugar_box/models/view_0_006_color.png",
        "../data/004_sugar_box/models/view_0_007_color.png",
        "../data/004_sugar_box/models/view_0_008_color.png",
        "../data/004_sugar_box/models/view_0_009_color.png",

        "../data/004_sugar_box/models/view_30_000_color.png",
        "../data/004_sugar_box/models/view_30_001_color.png",
        "../data/004_sugar_box/models/view_30_002_color.png",
        "../data/004_sugar_box/models/view_30_003_color.png",
        "../data/004_sugar_box/models/view_30_004_color.png",
        "../data/004_sugar_box/models/view_30_005_color.png",
        "../data/004_sugar_box/models/view_30_006_color.png",
        "../data/004_sugar_box/models/view_30_007_color.png",
        "../data/004_sugar_box/models/view_30_008_color.png",
        "../data/004_sugar_box/models/view_30_009_color.png",

        "../data/004_sugar_box/models/view_60_000_color.png",
        "../data/004_sugar_box/models/view_60_001_color.png",
        "../data/004_sugar_box/models/view_60_002_color.png",
        "../data/004_sugar_box/models/view_60_003_color.png",
        "../data/004_sugar_box/models/view_60_004_color.png",
        "../data/004_sugar_box/models/view_60_005_color.png",
        "../data/004_sugar_box/models/view_60_006_color.png",
        "../data/004_sugar_box/models/view_60_007_color.png",
        "../data/004_sugar_box/models/view_60_008_color.png",
        "../data/004_sugar_box/models/view_60_009_color.png",
    };
    std::string scene_path{"../data/004_sugar_box/test_images/4_0025_000065-color.jpg"};

    std::vector<Mat> models;
    for (const auto& p : models_paths) {
        models.push_back(imread(p, IMREAD_GRAYSCALE)); // Load as color to draw colored circles
    }
    Mat scene = imread(scene_path, IMREAD_GRAYSCALE); // Load as color

    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 0;
    Ptr<SURF> detector = SURF::create( minHessian );

    std::vector<std::vector<KeyPoint>> keypoints_models(models.size());
    std::vector<KeyPoint> keypoints_scene;

    std::vector<Mat> descriptors_models(models.size());
    Mat descriptors_scene;
    for (int i = 0; i < models.size(); i++) {
        detector->detectAndCompute(models[i], noArray(), keypoints_models[i], descriptors_models[i]);
    }
    detector->detectAndCompute(scene, noArray(), keypoints_scene, descriptors_scene);

    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector<std::vector< std::vector<DMatch> >> knn_matches(models.size());
    for (int i = 0; i < models.size(); i++) {
        matcher->knnMatch(descriptors_models[i], descriptors_scene, knn_matches[i], 2 );
    }

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.6;
    std::vector<std::vector<DMatch>> good_matches(models.size());
    for (int j = 0; j < knn_matches.size(); j++) {
        for (size_t i = 0; i < knn_matches[j].size(); i++) {
            if (knn_matches[j][i][0].distance < ratio_thresh * knn_matches[j][i][1].distance)
            {
                good_matches[j].push_back(knn_matches[j][i][0]);
            }
        }
    }

    //-- Draw good matches on the original images
    //Mat img1_matches = img1.clone();
    //Mat scene_matches = scene.clone();
    Mat scene_matches;
    cvtColor(scene, scene_matches, COLOR_GRAY2BGR);
    Scalar color = Scalar(0, 255, 0); // Green color for the circles
    int radius = 5;
    int thickness = 2;

    for (int i = 0; i < good_matches.size(); i++) {
        for (const auto& match : good_matches[i]) {
            //Point2f pt1 = keypoints1[match.queryIdx].pt;
            Point2f pt2 = keypoints_scene[match.trainIdx].pt;
            //circle(scene_matches, pt1, radius, color, thickness);
            circle(scene_matches, pt2, radius, color, thickness);
        }
    }

    //-- Show detected matches on original images
    //imshow("Good Matches on Image 1", scene_matches );
    imshow("Good Matches on Image 2", scene_matches);

    //-- Also show the combined matches for comparison
    //Mat img_matches;
    //drawMatches( img1, keypoints1, scene, keypoints2, good_matches, img_matches, Scalar::all(-1),
    //             Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //imshow("Combined Good Matches", img_matches );


    waitKey();
    return 0;
}
#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif
