// Include guards.
#ifndef PERFOMANCE_METRICS_H
#define PERFOMANCE_METRICS_H

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include <string.h>
#include <vector>
#include <fstream>
#include <string>


class PerformanceMetrics{

    public:
        // CONSTRUCTORS:
        PerformanceMetrics(const std::string &path_pred_labels, const std::string &path_true_labels) : path_pred_labels(path_pred_labels),
                                                                                                      path_true_labels(path_true_labels),
                                                                                                      sugar_p(2),
                                                                                                      mustard_p(2),
                                                                                                      power_drill_p(2),
                                                                                                      sugar_t(2),
                                                                                                      mustard_t(2),
                                                                                                      power_drill_t(2)
        {}

        // FUNCTION MEMBERS:
        double compute_detection_accuracy();
        
    private:
        // DATA MEMBER:
        // paths to file of predicted labels and true labels for which apply the metrics
        std::string path_pred_labels, path_true_labels; 

        std::vector<std::pair<int, int>> sugar_p, mustard_p, power_drill_p;
        std::vector<std::pair<int, int>> sugar_t, mustard_t, power_drill_t;
        // PRIVATE FUNCTION:
        
        // This function compute the mean intersection over union.
        double compute_mIoU();

};

// HELPER FUNCTIONS:
    // This function returns the two point <x_max, y_max>, <x_min, y_min> given a line.
    //std::vector<cv::Point>
    void parser(const std::string& path , std::vector<std::pair<int, int>>& sugar, std::vector<std::pair<int, int>>& mustard, std::vector<std::pair<int, int>>& power_drill);
    

    // Possible helper function to write in a text file the perfomances.

    // DEBUG:

    void printValue(std::vector<std::pair<int, int>> v1, std::vector<std::pair<int, int>> v2, std::vector<std::pair<int, int>> v3);

    




#endif