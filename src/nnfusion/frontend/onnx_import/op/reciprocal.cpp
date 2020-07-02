// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <memory>
#include <vector>

#include "ngraph/shape.hpp"
#include "nnfusion/core/operators/constant.hpp"
#include "nnfusion/core/operators/divide.hpp"

#include "utils/broadcasting.hpp"

#include "reciprocal.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector reciprocal(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);

                    std::shared_ptr<ngraph::Node> one_node = std::make_shared<ngraph::op::Constant>(
                        data->get_element_type(), Shape{}, std::vector<double>{1});
                    one_node = make_broadcast_node(one_node, data->get_shape());

                    return {one_node / data};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
