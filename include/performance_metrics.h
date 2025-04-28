// Authors: Chinello Alessandro, Piai Luca, Scantamburlo Mattia
// (Read the report)

#ifndef PERFORMANCE_METRICS_H
#define PERFORMANCE_METRICS_H

#include <vector>
#include <iomanip> // For std::fixed e std::setprecision
#include "opencv2/core/types.hpp"

class PerformanceMetrics{

    public:
        PerformanceMetrics(const std::string &path_pred_labels, const std::string &path_true_labels) 
            : path_pred_labels(path_pred_labels), path_true_labels(path_true_labels),
            sugar_p(2), mustard_p(2), power_drill_p(2), sugar_t(2), mustard_t(2),
            power_drill_t(2) { }

        // Function to write in a file and computing in terminal the metrics for the scenepath.
        void print_metrics();

    private:
        // Paths to file of predicted labels and true labels for which apply the metrics.
        std::string path_pred_labels, path_true_labels; 
        // Vectors in which memorize the coordinate of the read values from the label txt file.
        std::vector<cv:: Point2f> sugar_p, mustard_p, power_drill_p;
        std::vector<cv:: Point2f> sugar_t, mustard_t, power_drill_t;
        // IoU for sugar, mustard and power_drill computed comparing pred labels 
        // with true labels (with this specific sort/mapping).
        double IoU[3];
        // Initialize a 3vec of falses values to track if some object is missing in the SCENE.
        bool miss[3] = {false, false, false};

        // Function to store in the data members the IoU and the missing items
        void compute_IoU();

};

// Function to parse the label txt file with the predicted label of our algorithm.
void parser(const std::string& path , std::vector<cv:: Point2f>& sugar,
        std::vector<cv::Point2f>& mustard, std::vector<cv::Point2f>& power_drill);
// Debugging function.
void print_value(std::vector<cv::Point2f> v1, std::vector<cv::Point2f> v2, std::vector<cv::Point2f> v3);

#endif
