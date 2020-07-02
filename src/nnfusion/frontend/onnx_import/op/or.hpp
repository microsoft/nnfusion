// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "ngraph/node_vector.hpp"
#include "nnfusion/core/operators/or.hpp"

#include "core/node.hpp"
#include "utils/broadcasting.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline NodeVector logical_or(const Node& node)
                {
                    NodeVector ng_inputs{
                        numpy_style_broadcast_for_binary_operation(node.get_ng_inputs())};
                    return {std::make_shared<ngraph::op::Or>(ng_inputs.at(0), ng_inputs.at(1))};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
