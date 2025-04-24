
// Include guards.
#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include <string.h>

using namespace cv;

// GLOBAL CONSTANTS
    // Models images.
    const std::string models_paths[] = {
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

// HELPER FUNCTIONS:
    void draw_box(Mat& image);

#endif