#!/bin/bash

# OSF token is read from the environment -- never hardcode it.
#   export OSF_TOKEN=...   (a personal access token with write access)
osfToken="${OSF_TOKEN:?set OSF_TOKEN in the environment}"

osfJson="https://files.osf.io/v1/resources/vmdu6/providers/osfstorage/"
osfJpeg1="https://files.osf.io/v1/resources/tskx2/providers/osfstorage/"
osfJpeg2="https://files.osf.io/v1/resources/ksfjc/providers/osfstorage/"
dirjson="./json"
dirjpeg="./jpg"

# Create a tar file of all the files in the directory
tar -czf json.tar.gz -C "$dirjson" .

# Send the tar file via curl
response=$(curl -s -w "%{http_code}" -X PUT "${osfJson}?type=files&name=json.tar.gz" \
         -H "Content-Type: application/gzip" \
         -H "Authorization: Bearer ${osfToken}" \
         -T "json.tar.gz")
echo "json.tar.gz ${response}"

tar -czf jpg.tar.gz -C "$dirjpeg" .

# Split the tar file into multiple 1GB files
split -b 1G jpg.tar.gz jpg.tar.gz.part.

# Initialize counter
counter=1

# Loop over the split files and send each one via curl
for file in jpg.tar.gz.part.*
do
    # Choose the upload URL based on the counter
    if [ $counter -le 4 ]
    then
        uploadUrl=$osfJpeg1
    else
        uploadUrl=$osfJpeg2
    fi

    response=$(curl -s -w "%{http_code}" -X PUT "${uploadUrl}?type=files&name=${file}" \
             -H "Content-Type: application/gzip" \
             -H "Authorization: Bearer ${osfToken}" \
             -T "${file}")
    echo "${file} ${response}"

    # Increment counter
    ((counter++))
done


# Remove the tar files
rm json.tar.gz
rm jpg.tar.gz
rm jpg.tar.gz.part.*
