// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise absolute value operation.
        ///
        class Abs : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs an absolute value operation.
            ///
            Abs();
        };
    }
}
