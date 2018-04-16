@echo off
setlocal ENABLEDELAYEDEXPANSION

for %%V in (15,14,12,11) do if exist "!VS%%V0COMNTOOLS!" call "!VS%%V0COMNTOOLS!..\..\VC\vcvarsall.bat" amd64 && goto compile

:compile
nvcc -arch compute_30 src\ebsynth.cu -m64 -O6 -w -I "include" -o "bin\ebsynth.exe" -Xcompiler "/DNDEBUG /Ox /Oy /Gy /Oi /fp:fast" -Xlinker "/IMPLIB:\"lib\ebsynth.lib\"" || goto error
goto :EOF

:error
echo FAILED
@%COMSPEC% /C exit 1 >nul
