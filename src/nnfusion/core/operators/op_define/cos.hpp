// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise cosine operation.
        class Cos : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a cosine operation.
            Cos();
        };
    }
}
