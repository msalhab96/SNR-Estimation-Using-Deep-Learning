#!/bin/bash

base_url="https://www.ecs.utdallas.edu/loizou/speech/noizeus/"

files=(
    "train_0dB.zip" 
    "babble_0dB.zip" 
    "car_0dB.zip" 
    "exhibition_0dB.zip" 
    "restaurant_0dB.zip" 
    "street_0dB.zip" 
    "airport_0dB.zip" 
    "station_0dB.zip"
    )

temp_dir="noise-data-temp"

for file in ${files[@]}
    do
        full_url="${base_url}${file}"
        wget $full_url
        unzip -o $file -d $temp_dir
        rm $file
    done

data_dir="${temp_dir}/0dB"

mv "${data_dir}" "noise-data"

rm --recursive --force $temp_dir