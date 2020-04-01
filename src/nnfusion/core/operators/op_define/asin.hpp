// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise inverse sine (arcsin) operation.
        ///
        class Asin : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs an arcsin operation.
            Asin();
        };
    }
}
