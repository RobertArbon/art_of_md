#!/bin/bash

# Creates a gif from pngs. 
file_dir=../movies/fast2

cd $file_dir

# resize the PNGs if necessary
# mogrify -resize 512x512 $file_dir/*.png

# convert to jpgs if necessary
# magick mogrify -format jpg *.png

convert -delay 10 -loop 0 *.jpg output.gif

