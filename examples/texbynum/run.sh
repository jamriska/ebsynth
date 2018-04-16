#!/bin/sh
export PATH=../../bin:$PATH

ebsynth -patchsize 3 -uniformity 1000 -style source_photo.png -guide source_segment.png target_segment.png -output output.png

