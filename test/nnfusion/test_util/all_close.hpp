//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include "nnfusion/common/common.hpp"

union FloatUnion {
    float f;
    uint32_t i;
};

namespace nnfusion
{
    namespace test
    {
        /// \brief Same as numpy.allclose
        /// \param a First tensor to compare
        /// \param b Second tensor to compare
        /// \param rtol Relative tolerance
        /// \param atol Absolute tolerance
        /// \returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
        template <typename T>
        bool all_close(const std::vector<T>& a,
                       const std::vector<T>& b,
                       T rtol = static_cast<T>(1e-5),
                       T atol = static_cast<T>(1e-8))
        {
            bool rc = true;
            assert(a.size() == b.size());
            for (size_t i = 0; i < a.size(); ++i)
            {
                if (std::abs(a[i] - b[i]) > atol + rtol * std::abs(b[i]))
                {
                    NNFUSION_LOG(INFO) << a[i] << " is not close to " << b[i] << " at index " << i;
                    rc = false;
                }
            }
            return rc;
        }

        bool close_f(float a, float b, int mantissa_bits, int tolerance_bits);

        bool all_close_f(const std::vector<float>& a,
                         const std::vector<float>& b,
                         int mantissa_bits = 8,
                         int tolerance_bits = 2);

        bool all_close_f(const DataBuffer& a,
                         const DataBuffer& b,
                         int mantissa_bits = 8,
                         int tolerance_bits = 2);
    }
}
