// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/argmax.hpp"
#include "core/node.hpp"
#include "ngraph/node_vector.hpp"
#include "utils/reduction.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector argmax(const Node& node)
                {
                    return {reduction::make_ng_index_reduction_op<ngraph::op::ArgMax>(node)};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
