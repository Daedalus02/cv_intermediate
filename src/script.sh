#!/bin/bash

# Define the directory path
directory_path="../data/004_sugar_box/test_images/" # Replace with the actual path to your directory

# Check if the directory exists
if [ ! -d "$directory_path" ]; then
  echo "Error: Directory '$directory_path' not found."
  exit 1
fi

# Loop through each file in the directory
for file in "$directory_path"/*; do
  # Check if it's a regular file (not a directory, etc.)
  if [ -f "$file" ]; then
    # Execute the command with the file path
    ./a.out -s "../data/004_sugar_box/models" -p "../data/035_power_drill/models/" -m "../data/006_mustard_bottle/models" -i "$file"
    # You can add more commands here if needed
  fi
done

echo "Script finished."
