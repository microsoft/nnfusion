// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise hyperbolic tangent operation.
        class Tanh : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a hyperbolic tangent operation.
            Tanh();
        };
    }
}
