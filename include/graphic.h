#ifndef GRAPHIC
#define GRAPHIC

#include <opencv2/opencv.hpp>
#include <cmath> 
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

class graphic{

    public:
        // CONSTRUCTORS:
        graphic(const cv::Mat& source);
        // MEMBER FUNCTIONS:
        // This function is used to compute all the possible rotation of the template. 
        void rotateImage(double angle);
        // Getter function.
        const cv::Mat& getImage(){return image;}
        int cols(){return image.cols;}
        int rows(){return image.rows;}
        // Setter function.
        void setImage(const cv::Mat& source);
        // OPERATORS:
    private: 

        // DATA MEMBERS:
        cv::Mat image;


};


#endif