// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/max_pool.hpp"
#include "ngraph/node.hpp"
#include "ngraph/node_vector.hpp"

#include "utils/convpool.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector max_pool(const Node& node)
                {
                    return convpool::make_ng_pool<ngraph::op::MaxPool>(node);
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
