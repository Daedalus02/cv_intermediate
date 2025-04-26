#!/bin/bash

# Define the project root directory (now one level above 'build')
PROJECT_ROOT=$(dirname "$0")

# Define the directory containing the executable (the 'build' directory, which is the current directory)
EXECUTABLE_DIR="."

# Define the executable name
EXECUTABLE_NAME="obj_detector"

# Define the directory containing the input files (now one level above 'build')
INPUT_DIR="$PROJECT_ROOT/../data/006_mustard_bottle/test_images"
INPUT_DIR2="$PROJECT_ROOT/../data/006_mustard_bottle/labels"
MODELS_DRILL="$PROJECT_ROOT/../data/035_power_drill/models/"
MODELS_MUSTARD="$PROJECT_ROOT/../data/006_mustard_bottle/models/"
MODELS_SUGAR="$PROJECT_ROOT/../data/004_sugar_box/models/"

# Loop through each matching file
for filepath in "$INPUT_DIR"/*-color.jpg; do
    # Extract the filename from the full path
    filename=$(basename "$filepath")

    # Remove the '-color.jpg' suffix to get the base name
    base="${filename%-color.jpg}"

    # Construct the path to the modified label file
    modify_path="${base}-box.txt"

    # Execute the command with the relative path to the executable (now in the same directory)
    "$EXECUTABLE_DIR/$EXECUTABLE_NAME" \
        -p "$MODELS_DRILL" \
        -m "$MODELS_MUSTARD" \
        -s "$MODELS_SUGAR" \
        -i "$filepath" \
        -l "$INPUT_DIR2/$modify_path"
done