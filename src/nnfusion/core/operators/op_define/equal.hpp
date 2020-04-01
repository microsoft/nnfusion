// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/binary_elementwise_comparison.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise is-equal operation.
        class Equal : public BinaryElementwiseComparison
        {
        public:
            /// \brief Constructs an is-equal operation.

            Equal();
        };
    }
}
