// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"
#include "nnfusion/common/languageunit.hpp"

#include "cpu_kernelops.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            shared_ptr<LanguageUnit>
                get_eigen_math_kernel(const std::string& name,
                                      const std::string& math_kernel,
                                      size_t data_size,
                                      const std::vector<std::string>& data_types);

            shared_ptr<LanguageUnit>
                get_simd_math_kernel(const std::string& name,
                                     const std::string& math_kernel,
                                     size_t data_size,
                                     const std::vector<std::string>& data_types);
        }
    }
}
