#!/bin/sh
export PATH=../../bin:$PATH

ebsynth -style source_style.png \
        -guide source_fullgi.png target_fullgi.png -weight 0.5 \
        -guide source_dirdif.png target_dirdif.png -weight 0.5 \
        -guide source_dirspc.png target_dirspc.png -weight 0.5 \
        -guide source_indirb.png target_indirb.png -weight 0.5 \
        -output output.png
