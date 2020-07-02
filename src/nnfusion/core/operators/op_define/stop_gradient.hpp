// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief create StopGrdient op
        class StopGradient : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs StopGradient
            StopGradient();
        };
    }
}
