// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise square root operation.
        class Sqrt : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a square operation.
            Sqrt();
        };
    }
}
