// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise multiplication operation.
        class Multiply : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a multiplication operation.
            Multiply();

        protected:
            virtual bool is_commutative() override { return true; }
        };
    };
}
