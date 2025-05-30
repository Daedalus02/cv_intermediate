# Computer Vision Project: Object Detection and Feature Matching

## Project Overview
The goal of this project is to develop an object detection system capable of locating known objects ("power drill", "mustard bottle", "sugar box") in an input image. The system uses synthetic views to detect objects and highlights them with bounding boxes. The project is implemented in C++ using OpenCV, with a focus on robust feature matching or alternative approaches like template matching.

## Dataset
The dataset includes:
- `test_images`: Images for testing the system.
- `models`: 29 synthetic views of each object.
- `labels`: Ground truth annotations for object positions in the images.

Dataset link: [Google Drive](https://drive.google.com/drive/folders/1heXAbX4WKXf3-z2sl68Qg-cvbcVwosxO?usp=sharing)

## Outputs
- Bounding box coordinates saved in a text file.
- An image with bounding boxes superimposed on the original input.

## Performance Metrics
- **Mean Intersection over Union (mIoU)**: Average IoU for each object category.
- **Detection Accuracy**: Number of correctly detected objects (IoU > 0.5).

## Project Structure
project/

├── src/ Source code (C++ files)

├── data/ Dataset (test_images, models, labels)

├── include/ Declarations of the classes created

├── lib/ Definitions of the classes created

├── report/ Project report (PDF)

├── CMakeLists.txt

└── README.md This file

## Requirements
- C++ compiler (supporting C++11 or later)
- OpenCV library (version 4.x recommended)

## Build Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/Daedalus02/cv_intermediate
   cd cv_intermediate
2. Compile the code:
   ```bash
    mkdir build && cd build
    cmake .. && make
3. Run the executable:
   ```bash
   ./object_detector -s <path_to_sugar_box_models> -p <path_to_power_drill_models> -m <path_to_mustard_bottle_models>
      -i <path_to_test_images> -l <path_to_test_images_labels>
Example:
   ```bash
     ./obj_detector 
       -p ../data/035_power_drill/models 
       -m ../data/006_mustard_bottle/models 
       -s ../data/004_sugar_box/models 
       -i ../data/004_sugar_box/test_images/4_0001_000121-color.jpg 
       -l ../data/004_sugar_box/labels/4_0001_000121-box.txt 
   ``` 
   IMPORTANT NOTE: The models directories must not contain the masks but only the RGB views.
## Authors

- **Student 1** - [GitHub](https://github.com/Ale10chine)  
- **Student 2** - [GitHub](https://github.com/luca037)  
- **Student 3** - [GitHub](https://github.com/Daedalus02)  

## Report

The final report includes:

    Methodology and approach

    Performance metrics (mIoU and detection accuracy)

    Output images for all test cases

    Contributions and working hours per member
