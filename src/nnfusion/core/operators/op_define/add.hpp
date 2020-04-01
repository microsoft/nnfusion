// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise addition operation.
        ///
        class Add : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs an addition operation.
            ///
            Add();

        protected:
            virtual bool is_commutative() override { return true; }
        };
    }
}
