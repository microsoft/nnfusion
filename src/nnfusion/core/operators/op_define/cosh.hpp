// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise hyperbolic cosine (cosh) operation.
        class Cosh : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a hyperbolic cosine operation.
            Cosh();
        };
    }
}
