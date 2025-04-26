cmake_minimum_required(VERSION 3.10)
project(CV_Intermediate)

set(CMAKE_CXX_STANDARD 17)

# Instruct CMake to process the 'lib' subdirectory
add_subdirectory(lib)

# Find OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found (using pkg-config): ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS} include)
    set(OpenCV_LIBS ${OpenCV_LIBS})
else()
    message(STATUS "OpenCV not found via pkg-config, trying manual linking (Linux-like).")
    include_directories(/usr/include/opencv4 include)
    set(OpenCV_LIBS opencv_highgui opencv_core opencv_imgcodecs opencv_imgproc opencv_features2d opencv_xfeatures2d)
endif()

# Create the executable by compiling only test1.cpp from the 'src' directory
add_executable(cv_intermediate_app src/test1.cpp)

# Link the executable to the 'mylib' library created in 'lib'
target_link_libraries(cv_intermediate_app mylib ${OpenCV_LIBS})
