// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/util/binary_elementwise_comparison.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise not-equal operation.
        class NotEqual : public BinaryElementwiseComparison
        {
        public:
            /// \brief Constructs a not-equal operation.
            ///
            NotEqual();
        };
    }
}
