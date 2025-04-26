cmake_minimum_required(VERSION 3.10)

project(CV_Intermediate)

set(CMAKE_CXX_STANDARD 17)

project(MiddleProject)

include_directories(include)

add_library(MiddleProject
    include/utils.h
    include/features_extractor.h
    include/features_matcher.h
    include/performance_metrics.h
    lib/utils.cpp
    lib/features_extractor.cpp
    lib/performance_metrics.cpp
    )

add_executable(obj_detector
    src/main.cpp
    )

target_link_libraries(obj_detector MiddleProject)

find_package(OpenCV 4 REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV library found.")
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(obj_detector
        opencv_highgui
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        opencv_features2d
        )
else()
    message(FATAL_ERROR "OpenCV library not found.")
endif()