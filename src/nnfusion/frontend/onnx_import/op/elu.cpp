// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <memory>
#include <vector>

#include "ngraph/node.hpp"

#include "nnfusion/core/operators/add.hpp"
#include "nnfusion/core/operators/broadcast.hpp"
#include "nnfusion/core/operators/constant.hpp"
#include "nnfusion/core/operators/exp.hpp"
#include "nnfusion/core/operators/maximum.hpp"
#include "nnfusion/core/operators/minimum.hpp"
#include "nnfusion/core/operators/multiply.hpp"
#include "nnfusion/core/operators/subtract.hpp"

#include "utils/broadcasting.hpp"

#include "elu.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector elu(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    double alpha = node.get_attribute_value<double>("alpha", 1);

                    std::shared_ptr<ngraph::Node> alpha_node =
                        std::make_shared<ngraph::op::Constant>(
                            data->get_element_type(), Shape{}, std::vector<double>{alpha});
                    alpha_node = make_broadcast_node(alpha_node, data->get_shape());

                    std::shared_ptr<ngraph::Node> zero_node =
                        std::make_shared<ngraph::op::Constant>(
                            data->get_element_type(), Shape{}, std::vector<double>{0});
                    zero_node = make_broadcast_node(zero_node, data->get_shape());

                    return {std::make_shared<ngraph::op::Maximum>(data, zero_node) +
                            alpha_node *
                                std::make_shared<ngraph::op::Exp>(
                                    std::make_shared<ngraph::op::Minimum>(data, zero_node)) -
                            alpha_node};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
