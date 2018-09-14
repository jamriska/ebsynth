#!/bin/sh
g++ src/ebsynth.cpp src/ebsynth_cpu.cpp src/ebsynth_nocuda.cpp -DNDEBUG -O6 -fopenmp -I"include" -std=c++11 -o bin/ebsynth
