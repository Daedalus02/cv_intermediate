# cv_intermediate
Object detection and Feature Matching project developed by three computer engineering master students.

# Computer Vision Project: Object Detection and Feature Matching

## Project Overview
The goal of this project is to develop an object detection system capable of locating known objects ("power drill", "mustard bottle", "sugar box") in an input image. The system uses synthetic views and binary masks to detect objects and highlights them with bounding boxes. The project is implemented in C++ using OpenCV, with a focus on robust feature matching or alternative approaches like template matching.

## Dataset
The dataset includes:
- `test_images`: Images for testing the system.
- `models`: 60 synthetic views of each object and corresponding binary masks.
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
├
├── src/ # Source code (C++ files)
├
├── data/ # Dataset (test_images, models, labels)
├
├── output/ # Generated outputs (bounding box images and text files)
├
├── report/ # Project report (PDF)
├
└── README.md # This file

## Requirements
- C++ compiler (supporting C++11 or later)
- OpenCV library (version 4.x recommended)

## Build Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd project
2. Compile the code:
   ```bash
    mkdir build && cd build
    cmake .. && make
3. Run the executable:
      ```bash
   ./object_detector <path_to_input_image> <path_to_models> <output_directory>

## Authors

- **Student 1** - [GitHub](https://github.com/Ale10chine)  
- **Student 2** - [GitHub](https://github.com/luca037)  
- **Student 3** - [GitHub](https://github.com/Daedalus02)  

Report

The final report includes:

    Methodology and approach

    Performance metrics (mIoU and detection accuracy)

    Output images for all test cases

    Contributions and working hours per member

Notes

