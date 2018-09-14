@echo off
setlocal ENABLEDELAYEDEXPANSION

for %%V in (15,14,12,11) do if exist "!VS%%V0COMNTOOLS!" call "!VS%%V0COMNTOOLS!..\..\VC\vcvarsall.bat" x86 && goto compile

:compile
nvcc -m32 -arch compute_30 src\ebsynth.cpp src\ebsynth_cpu.cpp src\ebsynth_cuda.cu -DNDEBUG -O6 -I "include" -o "bin\ebsynth.exe" -Xcompiler "/openmp /fp:fast" -Xlinker "/IMPLIB:dummy.lib" -w || goto error
nvcc -m32 -arch compute_30 src\ebsynth.cpp src\ebsynth_cpu.cpp src\ebsynth_cuda.cu -DNDEBUG -O6 -I "include" -o "bin\ebsynth.dll" -Xcompiler "/openmp /fp:fast" -Xlinker "/IMPLIB:lib\ebsynth.lib" -shared -DEBSYNTH_API=__declspec(dllexport) -w || goto error
del dummy.lib;dummy.exp 2> NUL
goto :EOF

:error
echo FAILED
@%COMSPEC% /C exit 1 >nul
