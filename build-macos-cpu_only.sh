#!/bin/sh
clang++ src/ebsynth.cpp src/ebsynth_cpu.cpp src/ebsynth_nocuda.cpp -DNDEBUG -O3 -I"include" -std=c++11 -o bin/ebsynth
