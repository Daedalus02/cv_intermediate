#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/flann.hpp"
#include <iostream>
#include <vector>

using namespace cv;

int main(int argc, char * argv[]){
    
    //Loading the images 
    Mat model = imread("../../data/004_sugar_box/models/view_0_001_color.png");
    Mat scene = imread("../../data/004_sugar_box/models/view_0_004_color.png");
    if(model.empty() || scene.empty()){
        std::cout<<"Error: The image was not loaded correctly!"<<std::endl;
        return -1;
    }

    // Create SIFT detector
    Ptr<SIFT> siftPtr = SIFT::create();

    // Creating the vector to hold the detected keypoints
    std::vector<KeyPoint> keypoints_scene, keypoints_model;
    // Creating the vectors to hol the descriptors associated with the 
    // Keypoints.
    Mat descriptors_scene, descriptors_model;

    // This object can compare the descriptors of 2 different images.
    Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();

    // This vector holds the matches after the compcarison of the 2 images.  
    std::vector<DMatch> matches; 

    // Detecting and computing the keypoints and descriptors for both the scene and the model
    siftPtr->detectAndCompute(scene, noArray(), keypoints_scene, descriptors_scene);
    siftPtr->detectAndCompute(model, noArray(), keypoints_model, descriptors_model);

    // Checking whether there were some matches bethween the descriptors.
    matcher->match(descriptors_scene, descriptors_model, matches);

    // Sort matches by distance (best matches first)
    std::sort(matches.begin(), matches.end());

    // Take top N matches (percentage-based)
    const int numGoodMatches = matches.size() * 0.4;  // Take top 15%
    // Creating the vector with the best matches only.
    std::vector<DMatch> goodMatches(matches.begin(), matches.begin() + numGoodMatches);

    Mat imgMatches;

    // Printing an image composed by the scene and the model and painting 
    // the detected matches in it.
    drawMatches(
        scene, 
        keypoints_scene,
        model, 
        keypoints_model,
        goodMatches,
        imgMatches,
        Scalar::all(-1),  // Colore random per i match
        Scalar::all(-1),
        std::vector<char>(),
        DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );

    // Printing the compute result.
    namedWindow("Result");
    imshow("Result", imgMatches);
   
    /* 
    // Draw keypoints on the image
    Mat output;
    drawKeypoints(scene, keypoints_scene, output);
    namedWindow("Detected features");
    imshow("Detected features", output);

    //namedWindow("view_0_001_color.png");
    //imshow("view_0_001_color.png.png", img);*/
    waitKey(0);
    return 0;
}