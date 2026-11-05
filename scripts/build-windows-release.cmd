@echo off
rem Build the windows-release preset (LichtFeld-Studio.exe).
rem Usage: scripts\build-windows-release.cmd [clean]
rem
rem With Chinese MSVC, cl reports its /showIncludes notes in UTF-8 and the
rem compiler cache garbles them. Unless CMake generates under code page 65001
rem with the cache off, ninja records no header dependencies and header edits
rem never recompile their users.
setlocal

chcp 65001 >nul
for %%i in ("%~dp0..") do set "ROOT=%%~fi"
set "BUILD=%ROOT%\build-windows-release"

if defined VCINSTALLDIR goto :have_msvc
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSROOT="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%i"
if not defined VSROOT (
    echo Visual Studio with the C++ x64 tools was not found.
    exit /b 1
)
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
:have_msvc

rem --------------------------------------------------------------------------
rem Toolchain / vcpkg setup for this machine.
rem vcpkg manifest mode requires:
rem   * cmake >= 4.4.4 (toolchain requirement)
rem   * PowerShell 7 (for the overlay ports' meson build system)
rem   * 7-Zip (for source archive extraction)
rem   * CUDA Toolkit + MSVC environment (already loaded by vcvars64 above)
rem --------------------------------------------------------------------------
set "VCPKG_ROOT=D:\kmf_soft\vcpkg"
set "VCPKG_FORCE_SYSTEM_BINARIES=1"

rem Put vcpkg downloads on the proxy (vcpkg auto-detects IE proxy, but the
rem GitHub cert error from earlier runs suggests HTTPS_PROXY was set with an
rem https:// prefix; forcing http:// makes curl happy).
if not defined HTTP_PROXY  set "HTTP_PROXY=http://127.0.0.1:7897"
if not defined HTTPS_PROXY set "HTTPS_PROXY=http://127.0.0.1:7897"

rem Prepend our pinned toolchain directories to PATH (checked one by one so
rem the script still works on machines without this exact layout).
set "PATH=%PATH%;D:\kmf_soft\poweshell\7;D:\kmf_soft\tools\ninja;D:\kmf_soft\tools\cmake4\bin"
if exist "C:\Program Files\7-Zip\7z.exe"  set "PATH=C:\Program Files\7-Zip;%PATH%"

rem Pick cmake.exe: pinned 4.4.x first, fall back to whatever is on PATH.
set "CMAKE=cmake"
if exist "D:\kmf_soft\tools\cmake4\bin\cmake.exe" set "CMAKE=D:\kmf_soft\tools\cmake4\bin\cmake.exe"

rem Clean before configuring: clean also deletes the version .rc files that configure generates.
if /i not "%~1"=="clean" goto :configure
"%CMAKE%" --build "%BUILD%" --target clean
if errorlevel 1 exit /b 1

:configure
pushd "%ROOT%"
"%CMAKE%" --preset windows-release -DENABLE_COMPILER_CACHE=OFF
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" exit /b %RC%

"%CMAKE%" --build "%BUILD%" --target LichtFeld-Studio
if errorlevel 1 exit /b 1

echo.
echo Built %BUILD%\LichtFeld-Studio.exe
