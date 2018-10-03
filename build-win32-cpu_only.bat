@echo off
setlocal ENABLEDELAYEDEXPANSION

call "vcvarsall.bat" x86

cl src\ebsynth.cpp src\ebsynth_cpu.cpp src\ebsynth_nocuda.cpp /DNDEBUG /O2 /openmp /EHsc /nologo /I"include" /Fe"bin\ebsynth.exe" || goto error
cl src\ebsynth.cpp src\ebsynth_cpu.cpp src\ebsynth_nocuda.cpp /DNDEBUG /O2 /openmp /EHsc /nologo /I"include" /Fe"bin\ebsynth.dll" /DEBSYNTH_API="__declspec(dllexport)" /link /IMPLIB:"lib\ebsynth.lib" || goto error
del ebsynth.obj;ebsynth_cpu.obj;ebsynth_nocuda.obj 2> NUL
goto :EOF

:error
echo FAILED
@%COMSPEC% /C exit 1 >nul
