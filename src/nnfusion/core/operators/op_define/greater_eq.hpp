// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/binary_elementwise_comparison.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise greater-than-or-equal operation.
        class GreaterEq : public BinaryElementwiseComparison
        {
        public:
            /// \brief Constructs a greater-than-or-equal operation.
            GreaterEq();
        };
    }
}
