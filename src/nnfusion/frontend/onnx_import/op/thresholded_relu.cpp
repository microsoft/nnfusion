// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <memory>
#include <vector>

#include "ngraph/node.hpp"

#include "nnfusion/core/operators/broadcast.hpp"
#include "nnfusion/core/operators/constant.hpp"
#include "nnfusion/core/operators/convert.hpp"
#include "nnfusion/core/operators/greater.hpp"
#include "nnfusion/core/operators/multiply.hpp"

#include "core/node.hpp"
#include "utils/broadcasting.hpp"

#include "thresholded_relu.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector thresholded_relu(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    double alpha = node.get_attribute_value<double>("alpha", 1.0);

                    std::shared_ptr<ngraph::Node> alpha_node =
                        std::make_shared<ngraph::op::Constant>(
                            data->get_element_type(), ngraph::Shape{}, std::vector<double>{alpha});
                    alpha_node = make_broadcast_node(alpha_node, data->get_shape());

                    auto data_map = std::make_shared<ngraph::op::Convert>(
                        std::make_shared<ngraph::op::Greater>(data, alpha_node),
                        data->get_element_type());
                    return {data * data_map};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
