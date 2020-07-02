// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/core/operators/op.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Generic constant-padding operation.
        class NoOp : public Op
        {
        public:
            NoOp(std::string name)
                : Op(name)
            {
            }
        };
    }
}
