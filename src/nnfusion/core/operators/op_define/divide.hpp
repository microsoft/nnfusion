// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise division operation.
        class Divide : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a division operation.
            Divide();
        };

        /// \brief Elementwise division operation.
        class DivNoNan : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a division operation.
            DivNoNan();
        };
    }
}
