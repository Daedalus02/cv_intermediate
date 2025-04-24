#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include <string.h>
#include "../include/features_extractor.h"

/*FeaturesExctractor::FeaturesExctractor(std::string detector_type){
    swicth(detector_type){
        case "SIFT":
            detector = SIFT::create();
        case "SURF":
            detector = SURF::create();
        default:
            detector = SIFT::create();
    }
}*/