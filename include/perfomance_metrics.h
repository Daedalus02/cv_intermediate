
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

using namespace cv;

class PerfomanceMetrics{

    public:
        // CONSTRUCTORS:
        PerfomanceMetrics();

        // FUNCTION MEMBERS:
        double compute_detection_accuracy();
        
    private:
        // DATA MEMBER:

        // PRIVATE FUNCTION:
        
        // This function compute the mean intersection over union.
        double compute_mIoU();

        // Possible helper function to write in a text file the perfomances.

};

// HELPER FUNCTIONS:
    // This function returns the two point <x_max, y_max>, <x_min, y_min> given a line.
    std::vector<cv::Point> parser(std::string line);

#endif