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
            LU_DECLARE(thread);
            LU_DECLARE(eigen_tensor);
            LU_DECLARE(eigen_utils);
            LU_DECLARE(eigen_spatial_convolution);
            LU_DECLARE(cblas);
            LU_DECLARE(mlas);
            LU_DECLARE(threadpool);
            LU_DECLARE(barrier);
            LU_DECLARE(simd);
        }

        namespace macro
        {
        }

        namespace declaration
        {
            LU_DECLARE(eigen_global_thread_pool);
            LU_DECLARE(eigen_global_thread_pool_device);
            LU_DECLARE(worker_thread_pool);
            LU_DECLARE(schedule_thread_pool);
            LU_DECLARE(superscaler_schedule_thread);
        }
    } // namespace kernels
} // namespace nnfusion
