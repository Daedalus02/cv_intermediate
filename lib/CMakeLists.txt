cmake_minimum_required(VERSION 3.10)

add_library(mylib STATIC
    features_extractor.cpp
    performance_metrics.cpp
    utils.cpp
)

# (Optional) Specify the include directories needed to compile the library
target_include_directories(mylib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../include)