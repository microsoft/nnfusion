// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise subtraction operation.
        class Subtract : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs an subtraction operation.
            Subtract();
        };
    }
}
