// Include guards.
#ifndef PERFORMANCE_METRICS_H
#define PERFORMANCE_METRICS_H

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
        // Function to write in a file and computing in terminal the metrix for the scenepath CAMBIARE IN VOID in modo cche cosi crei in automatico il file e scriva per loutput delle metriche
        double print_metrics();

    private:
        // DATA MEMBER:
        // paths to file of predicted labels and true labels for which apply the metrics
        std::string path_pred_labels, path_true_labels; 
        // IoU for sugar, mustard and power_drill computed comparing pred labels with true labels
        double IoU[3];
        // Initialize a 3vec of falses values to track if some object is missing in the SCENE
        bool miss[3] = {false, false, false};

        std::vector<cv:: Point2f> sugar_p, mustard_p, power_drill_p;
        std::vector<cv:: Point2f> sugar_t, mustard_t, power_drill_t;
        // PRIVATE FUNCTION:
        // Function to store in the data members the IoU and the missing items
        void compute_IoU();

};

// HELPER FUNCTIONS:
    // This function returns the two point <x_max, y_max>, <x_min, y_min> given a line.
    //std::vector<cv::Point>
    void parser(const std::string& path , std::vector<cv:: Point2f>& sugar, std::vector<cv:: Point2f>& mustard, std::vector<cv:: Point2f>& power_drill);
    // DEBUG:
    void print_value(std::vector<cv:: Point2f> v1, std::vector<cv:: Point2f> v2, std::vector<cv:: Point2f> v3);


    




#endif