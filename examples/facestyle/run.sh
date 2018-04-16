#!/bin/sh
export PATH=../../bin:$PATH

ebsynth -style source_painting.png \
        -guide source_Gapp.png target_Gapp.png -weight 0.5 \
        -guide source_Gseg.png target_Gseg.png -weight 1.5 \
        -guide source_Gpos.png target_Gpos.png -weight 10 \
        -output output.png
