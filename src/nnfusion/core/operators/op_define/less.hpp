// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/core/operators/util/binary_elementwise_comparison.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise less-than operation.
        class Less : public BinaryElementwiseComparison
        {
        public:
            /// \brief Constructs a less-than operation.
            ///
            Less();
        };
    }
}
