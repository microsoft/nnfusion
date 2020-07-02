// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise floor operation.
        class Floor : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a floor operation.
            Floor();
        };
    }
}
