// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "nnfusion/core/kernels/common_langunit.hpp"

namespace nnfusion
{
    namespace codegen
    {
        namespace cmake
        {
            LU_DECLARE(cblas);
            LU_DECLARE(eigen);
            LU_DECLARE(mlas);
            LU_DECLARE(threadpool);
            LU_DECLARE(threads);
            LU_DECLARE(superscaler_cuda);
            LU_DECLARE(superscaler_rocm);
            LU_DECLARE(cuda_lib);
            LU_DECLARE(rocm_lib);
            LU_DECLARE(cub);
            LU_DECLARE(cutlass);
        } // namespace cmake
        namespace helper
        {
            LU_DECLARE(debug);
            LU_DECLARE(cuda_half_debug);
        } // namespace helper
    }     // namespace codegen
} // namespace nnfusion
