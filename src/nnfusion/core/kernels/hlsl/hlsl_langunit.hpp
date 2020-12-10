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

        } // namespace header

        namespace macro
        {
            // LU_DECLARE(CUDA_SAFE_CALL_NO_THROW);

        } // namespace macro

        namespace declaration
        {
            LU_DECLARE(antares_hlsl_dll);
        } // namespace declaration
    }     // namespace kernels
} // namespace nnfusion
