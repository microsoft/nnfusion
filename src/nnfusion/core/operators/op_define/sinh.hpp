// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise hyperbolic sine (sinh) operation.
        class Sinh : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a hyperbolic sine operation.
            Sinh();
        };
    }
}
