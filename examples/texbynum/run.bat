@echo off
setlocal
set PATH=..\..\bin;%PATH%

ebsynth.exe -patchsize 3 -uniformity 1000 -style source_photo.png -guide source_segment.png target_segment.png -output output.png
