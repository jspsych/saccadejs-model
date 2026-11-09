#!/bin/bash

input_folder="webm"
output_folder="jpg"
log_file="conversion_results.log"

# Create output folder if it doesn't exist
mkdir -p "$output_folder"
echo "Starting specialized conversion at $(date)" > "$log_file"

for file in "$input_folder"/*.webm; do
  filename=$(basename "$file")
  jpg_file="$output_folder/${filename%.webm}.jpg"

  if [ -f "$jpg_file" ]; then
    echo "Skipping existing: $jpg_file"
    continue
  fi

  echo "Processing $filename..." | tee -a "$log_file"

  # Method 1: Use FFmpeg rawvideo decoder which bypasses swscaler for color space
  ffmpeg -y -i "$file" -vframes 1 -c:v rawvideo -f rawvideo -pix_fmt yuv420p \
    "$output_folder/temp.yuv"

  if [ -f "$output_folder/temp.yuv" ] && [ -s "$output_folder/temp.yuv" ]; then
    # Convert raw YUV to JPEG (with new color parameters)
    ffmpeg -y -f rawvideo -pixel_format yuv420p -video_size 640x480 -i "$output_folder/temp.yuv" \
      -c:v mjpeg -qscale:v 2 -pix_fmt yuvj420p "$jpg_file"
    rm "$output_folder/temp.yuv"
  else
    # Method 2: If raw approach failed, try with pixfmt filter
    echo "  Trying method 2..." | tee -a "$log_file"
    ffmpeg -y -i "$file" -vframes 1 -vf "zscale=matrixin=709:all=bt709" \
      -c:v mjpeg -qscale:v 2 "$jpg_file" 2>> "$log_file"
  fi

  # Final verification
  if [ -f "$jpg_file" ] && [ -s "$jpg_file" ]; then
    echo "  Success: $jpg_file created" | tee -a "$log_file"
  else
    echo "  FAILED: Could not create $jpg_file" | tee -a "$log_file"
  fi
done

echo "Conversion completed at $(date)" >> "$log_file"
