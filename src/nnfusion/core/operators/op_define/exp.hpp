// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise natural exponential (exp) operation.
        class Exp : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs an exponential operation.
            ///
            Exp();
        };
    }
}
