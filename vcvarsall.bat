@echo off

for /f "usebackq tokens=*" %%i in (`vswhere -latest -legacy -property installationPath`)    do (set vsdir=%%i)
for /f "usebackq tokens=*" %%i in (`vswhere -latest -legacy -property installationVersion`) do (set vsver=%%i)

if %vsver% geq 15 (
  set vcvarsall="%vsdir%\VC\Auxiliary\Build\vcvarsall.bat"
) else (
  set vcvarsall="%vsdir%\VC\vcvarsall.bat"
)

echo %vcvarsall%

if exist %vcvarsall% (
  call %vcvarsall% %*
)
