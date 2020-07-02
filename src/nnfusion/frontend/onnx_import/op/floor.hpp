// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "ngraph/node_vector.hpp"
#include "nnfusion/core/operators/floor.hpp"

#include "core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline NodeVector floor(const Node& node)
                {
                    return {std::make_shared<ngraph::op::Floor>(node.get_ng_inputs().at(0))};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
