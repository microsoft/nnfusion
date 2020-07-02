// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "ngraph/node_vector.hpp"
#include "nnfusion/core/operators/negative.hpp"

#include "core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline NodeVector neg(const Node& node) { return {-node.get_ng_inputs().at(0)}; }
            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
