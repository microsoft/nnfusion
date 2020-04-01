// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        class Sigmoid : public ElementwiseArithmetic
        {
        public:
            Sigmoid();
        };

        /// \brief Elementwise SigmoidBackprop operation.
        class SigmoidBackprop : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a SigmoidBackprop operation.
            SigmoidBackprop();
        };
    }
}
