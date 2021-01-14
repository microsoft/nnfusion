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
            // LU_DECLARE(CUDA_SAFE_CALL_NO_THROW);

        } // namespace macro

        namespace declaration
        {
            LU_DECLARE(antares_hlsl_dll);
            LU_DECLARE(antares_hlsl_dll_cpp);
        } // namespace declaration
    }     // namespace kernels
} // namespace nnfusion
