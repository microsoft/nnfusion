// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise hyperbolic tangent operation.
        class Tanh : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a hyperbolic tangent operation.
            Tanh();
        };
    }
}
