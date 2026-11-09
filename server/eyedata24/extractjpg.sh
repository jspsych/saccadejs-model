#!/bin/bash

input_folder="webm"
output_folder="jpg"
log_file="conversion_log.txt"

# Create output folder if it doesn't exist
mkdir -p "$output_folder"
echo "Starting conversion at $(date)" > "$log_file"

for file in "$input_folder"/*.webm; do
  filename=$(basename "$file")
  jpg_file="$output_folder/${filename%.webm}.jpg"

  if [ -f "$jpg_file" ]; then
    echo "JPG file already exists: $jpg_file. Skipping."
  else
    echo "Processing: $filename"
    echo "Processing: $filename" >> "$log_file"

    # Method 1: Simplest approach - just extract first frame
    echo "  Trying method 1 (basic extraction)..." | tee -a "$log_file"
    ffmpeg -y -v error -i "$file" -vframes 1 "$jpg_file" 2>> "$log_file"

    # Method 2: If that fails, try with pixel format conversion
    if [ ! -f "$jpg_file" ] || [ ! -s "$jpg_file" ]; then
      echo "  Method 1 failed. Trying method 2 (pixel format)..." | tee -a "$log_file"
      ffmpeg -y -v error -i "$file" -vframes 1 -pix_fmt yuvj420p "$jpg_file" 2>> "$log_file"
    fi

    # Method 3: If that fails, try with explicit scale filter
    if [ ! -f "$jpg_file" ] || [ ! -s "$jpg_file" ]; then
      echo "  Method 2 failed. Trying method 3 (scale filter)..." | tee -a "$log_file"
      ffmpeg -y -v error -i "$file" -vframes 1 -vf "scale=640:480" "$jpg_file" 2>> "$log_file"
    fi

    # Method 4: If that fails, try with PNG intermediate format
    if [ ! -f "$jpg_file" ] || [ ! -s "$jpg_file" ]; then
      echo "  Method 3 failed. Trying method 4 (PNG intermediate)..." | tee -a "$log_file"
      ffmpeg -y -v error -i "$file" -vframes 1 "$output_folder/temp.png" 2>> "$log_file"
      ffmpeg -y -v error -i "$output_folder/temp.png" "$jpg_file" 2>> "$log_file"
      rm -f "$output_folder/temp.png"
    fi

    # Method 5: Last resort - try with full color space override
    if [ ! -f "$jpg_file" ] || [ ! -s "$jpg_file" ]; then
      echo "  Method 4 failed. Trying method 5 (color space override)..." | tee -a "$log_file"
      ffmpeg -y -v error -i "$file" -vframes 1 -vf "format=yuv420p" \
        -color_primaries bt709 -color_trc bt709 -colorspace bt709 "$jpg_file" 2>> "$log_file"
    fi

    # Verify final result
    if [ -f "$jpg_file" ] && [ -s "$jpg_file" ]; then
      echo "  Success: Created $jpg_file" | tee -a "$log_file"
    else
      echo "  ERROR: All methods failed for $filename" | tee -a "$log_file"
    fi
  fi
done

echo "All conversions completed at $(date)" >> "$log_file"
echo "Conversion complete. See $log_file for details."
