#!/bin/sh
nvcc -arch compute_30 src/ebsynth.cpp src/ebsynth_cpu.cpp src/ebsynth_cuda.cu -I"include" -DNDEBUG -D__CORRECT_ISO_CPP11_MATH_H_PROTO -O6 -std=c++11 -w -Xcompiler -fopenmp -o bin/ebsynth
