// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "nnfusion/core/kernels/common_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace header
        {
            LU_DECLARE(systems);
            LU_DECLARE(D3D12APIWrapper);

        } // namespace header

        namespace macro
        {
            LU_DECLARE(OutputDebugStringA);
            LU_DECLARE(RUNTIME_API);

        } // namespace macro

        namespace declaration
        {
            LU_DECLARE(antares_hlsl_dll_cs);
            LU_DECLARE(antares_hlsl_dll_cpp);
            LU_DECLARE(dxModuleLaunchAsync);

        } // namespace declaration
    }     // namespace kernels
} // namespace nnfusion
