@echo off
setlocal ENABLEDELAYEDEXPANSION

call "vcvarsall.bat" amd64

nvcc -arch compute_30 src\ebsynth.cpp src\ebsynth_cpu.cpp src\ebsynth_cuda.cu -DNDEBUG -O6 -I "include" -o "bin\ebsynth.exe" -Xcompiler "/openmp /fp:fast" -Xlinker "/IMPLIB:dummy.lib" -w || goto error
nvcc -arch compute_30 src\ebsynth.cpp src\ebsynth_cpu.cpp src\ebsynth_cuda.cu -DNDEBUG -O6 -I "include" -o "bin\ebsynth.dll" -Xcompiler "/openmp /fp:fast" -Xlinker "/IMPLIB:lib\ebsynth.lib" -shared -DEBSYNTH_API=__declspec(dllexport) -w || goto error
del dummy.lib;dummy.exp 2> NUL
goto :EOF

:error
echo FAILED
@%COMSPEC% /C exit 1 >nul
