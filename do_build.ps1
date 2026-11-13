param([string]$Target = 'LichtFeld-Studio')
& 'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64 -HostArch amd64
$env:PATH = 'D:\kmf_soft\tools\cmake4\bin;' + $env:PATH
$env:VCPKG_FORCE_SYSTEM_BINARIES = '1'
Set-Location 'D:\kmf_soft\LichtFeld-Studio'
cmake --build 'D:\kmf_soft\LichtFeld-Studio\build-windows-release' --target $Target --config Release -j 4 2>&1
