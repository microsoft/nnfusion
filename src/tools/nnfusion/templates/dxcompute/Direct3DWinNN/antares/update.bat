@echo off

curl -LOs https://raw.githubusercontent.com/microsoft/antares/v0.3.x/backends/c-hlsl_xbox/evaluator/AntaresHlslLib/D3D12APIWrapper.h && echo updated D3D12APIWrapper.h

curl -LOs https://raw.githubusercontent.com/microsoft/antares/v0.3.x/backends/c-hlsl_xbox/evaluator/AntaresHlslLib/D3D12APIWrapper.cpp && echo updated D3D12APIWrapper.cpp

curl -LOs https://raw.githubusercontent.com/microsoft/antares/v0.3.x/backends/c-hlsl_xbox/evaluator/AntaresHlslLib/D3D12Util.h && echo updated D3D12Util.h

echo finished!
pause
