#!/bin/sh
nvcc -arch compute_30 src/ebsynth.cu -o bin/ebsynth -I "include" -std=c++11 -Xcompiler "-DNDEBUG -O6 -D__CORRECT_ISO_CPP11_MATH_H_PROTO"
