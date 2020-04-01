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
                    LOG(INFO) << a[i] << " is not close to " << b[i];
                    rc = false;
                }
            }
            return rc;
        }
    }
}
