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

// Microsoft (c) 2019 X-mas, MSRA NNFusion Team
#include "all_close.hpp"

namespace nnfusion
{
    namespace test
    {
        bool close_f(float a, float b, int mantissa_bits, int tolerance_bits)
        {
            // isfinite(a) => !isinf(a) && !isnan(a)
            if (!isfinite(a) || !isfinite(b))
            {
                return false;
            }

            FloatUnion a_fu{a};
            FloatUnion b_fu{b};
            uint32_t a_uint = a_fu.i;
            uint32_t b_uint = b_fu.i;

            // A trick to handle both positive and negative numbers, see https://goo.gl/YbdnFQ
            // - If negative: convert to two's complement
            // - If positive: mask with sign bit
            uint32_t sign_mask = static_cast<uint32_t>(1U) << 31;
            a_uint = (sign_mask & a_uint) ? (~a_uint + 1) : (sign_mask | a_uint);
            b_uint = (sign_mask & b_uint) ? (~b_uint + 1) : (sign_mask | b_uint);

            uint32_t distance = (a_uint >= b_uint) ? (a_uint - b_uint) : (b_uint - a_uint);

            // e.g. for float with 24 bit mantissa, 2 bit accuracy, and hard-coded 8 bit exponent_bits
            // tolerance_bit_shift = 32 -           (1 +  8 + (24 -     1         ) - 2             )
            //                       float_length    sign exp  mantissa implicit 1    tolerance_bits
            uint32_t tolerance_bit_shift = 32 - (1 + 8 + (mantissa_bits - 1) - tolerance_bits);
            uint32_t tolerance = static_cast<uint32_t>(1U) << tolerance_bit_shift;

            return distance <= tolerance;
        }

        bool all_close_f(const std::vector<float>& a,
                         const std::vector<float>& b,
                         int mantissa_bits,
                         int tolerance_bits)
        {
            bool rc = true;
            if (a.size() != b.size())
            {
                throw nnfusion::errors::RuntimeError(
                    "a.size() != b.size() for all_close comparison.");
            }
            for (size_t i = 0; i < a.size(); ++i)
            {
                bool is_close_f = close_f(a[i], b[i], mantissa_bits, tolerance_bits);
                if (!is_close_f)
                {
                    NNFUSION_LOG(INFO) << a[i] << " is not close to " << b[i];
                    rc = false;
                }
            }
            return rc;
        }

        bool all_close_f(const DataBuffer& a,
                         const DataBuffer& b,
                         int mantissa_bits,
                         int tolerance_bits)
        {
            NNFUSION_CHECK(a.get_type() == element::f32 && b.get_type() == element::f32)
                << "element type of DataBuffer should be element::f32";
            bool rc = true;
            if (a.size() != b.size())
            {
                throw nnfusion::errors::RuntimeError(
                    "a.size() != b.size() for all_close comparison.");
            }
            for (size_t i = 0; i < a.size(); ++i)
            {
                float lhs, rhs;
                a.getElement(i, &lhs);
                b.getElement(i, &rhs);
                bool is_close_f = close_f(lhs, rhs, mantissa_bits, tolerance_bits);
                if (!is_close_f)
                {
                    NNFUSION_LOG(INFO) << lhs << " is not close to " << rhs;
                    rc = false;
                }
            }
            return rc;
        }
    }
}
