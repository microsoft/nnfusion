// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise maximum operation.
        class Maximum : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a maximum operation.
            Maximum();

            virtual bool is_commutative() override { return true; }
        };
    }
}
