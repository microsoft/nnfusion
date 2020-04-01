// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise sine operation.
        class Sin : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a sine operation.
            Sin();
        };
    }
}
