@echo off

echo Downloading dependencies ..
curl -LOs https://github.com/microsoft/antares/raw/library/antares_hlsl_v0.1_x64.dll

echo Compiling nnfusion_rt ..
C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe nnfusion_rt.cs

echo Compiling finished!
pause

echo Executing program ..
nnfusion_rt

echo Program finished!
pause
