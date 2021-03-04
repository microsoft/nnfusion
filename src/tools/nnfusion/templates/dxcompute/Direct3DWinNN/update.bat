@echo off

echo update D3D12APIWrapper.h
curl -LOs https://github.com/microsoft/antares/blob/master/backends/c-hlsl/evaluator/AntaresHlslLib/D3D12APIWrapper.h

echo update D3D12APIWrapper.cpp
curl -LOs https://github.com/microsoft/antares/blob/master/backends/c-hlsl/evaluator/AntaresHlslLib/D3D12APIWrapper.cpp

echo update D3D12Antares.h
curl -LOs https://github.com/microsoft/antares/blob/master/backends/c-hlsl/evaluator/AntaresHlslLib/D3D12Antares.h

echo finished!
pause
