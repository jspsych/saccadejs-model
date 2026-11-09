#!/bin/bash

input_folder="webm"
output_folder="jpg"

for file in "$input_folder"/*.webm; do
  filename=$(basename "$file")
  jpg_file="$output_folder/${filename%.webm}.jpg"

  if [ -f "$jpg_file" ]; then
    echo "JPG file already exists: $jpg_file. Skipping ffmpeg command."
  else
    echo "JPG file does not exist: $jpg_file. Running ffmpeg command."
    ffmpeg -y -i "$file" -update 1 -frames:v 1 "$jpg_file"
  fi
done
