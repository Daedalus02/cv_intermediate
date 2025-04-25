#!/bin/bash

# Define the directory containing the input files
INPUT_DIR="../data/004_sugar_box/test_images/"
INPUT_DIR2="../data/004_sugar_box/labels/"

# Loop through each matching file
for filepath in "$INPUT_DIR"/*-color.jpg; do
    # Extract the filename from the full path
    filename=$(basename "$filepath")
    #echo $filename

    # Remove the '-color.jpg' suffix to get the base name
    base="${filename%-color.jpg}"

    # Construct the <modify path>
    modify_path="${base}-box.txt"

    #echo $modify_path

    # Execute the command with constructed paths
    ./a.out -p ../data/035_power_drill/models -m ../data/006_mustard_bottle/models -s ../data/004_sugar_box/models -i "$filepath" color.jpg -l "$INPUT_DIR2/$modify_path"
done