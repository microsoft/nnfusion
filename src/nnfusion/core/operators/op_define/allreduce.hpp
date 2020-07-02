// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "../op.hpp"

namespace nnfusion
{
    namespace op
    {
        class AllReduce : public Op
        {
        public:
            AllReduce();

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;
        };
    }
}
