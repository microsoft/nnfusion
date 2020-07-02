// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "ngraph/node.hpp"
#include "ngraph/node_vector.hpp"
#include "nnfusion/core/operators/avg_pool.hpp"

#include "utils/convpool.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector global_average_pool(const Node& node)
                {
                    return convpool::make_ng_pool<ngraph::op::AvgPool>(node);
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
