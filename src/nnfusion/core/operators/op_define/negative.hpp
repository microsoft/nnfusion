// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise negative operation.
        class Negative : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a negative operation.
            Negative();
        };
    }
}
