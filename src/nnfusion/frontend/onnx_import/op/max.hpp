// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "ngraph/node_vector.hpp"
#include "nnfusion/core/operators/maximum.hpp"

#include "core/node.hpp"
#include "utils/variadic.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline NodeVector max(const Node& node)
                {
                    return variadic::make_ng_variadic_op<ngraph::op::Maximum>(node);
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
