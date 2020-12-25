//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "backward_registry.hpp"
#include "nnfusion/core/graph/util/autobroadcast.hpp"

REGISTER_BACKWARD_TRANSLATOR(Power).translator([](std::shared_ptr<GNode> forward_node,
                                                  const GNodeIndexVector& outputs_grad,
                                                  std::shared_ptr<nnfusion::graph::Graph> graph)
                                                   -> GNodeIndexVector {
    NNFUSION_CHECK(outputs_grad.size() == 1) << "power have only 1 output, but "
                                             << outputs_grad.size() << " outputs_grad provided";
    // z = x**y, x_grad = z_grad * y * x ** (y - 1) = z_grad * y * z / x, y_grad = z_grad * x ** y * lnx = z_grad * z * lnx
    // update: cannot computing x_grad by z_grad * y * z / x, because it returns nan for where x=0, it's the same for y_grad
    // we should implement a dedicate power_backward op
    NNFUSION_LOG(NNFUSION_WARNING) << "Current power backward might lead nan";
    auto x = get_node_input(forward_node, 0);
    auto y = get_node_input(forward_node, 1);
    auto z = get_node_output(forward_node, 0);

    auto base_deri = graph->add_node_and_edge(
        std::make_shared<op::GenericOp>(
            forward_node->get_name() + "_base_deri", "PowerBackwardBase", op::OpConfig::any()),
        {x, y});
    auto x_grad = graph->add_node_and_edge(std::make_shared<op::Multiply>(),
                                           {outputs_grad[0], GNodeIndex{base_deri, 0}});

    auto exp_deri = graph->add_node_and_edge(
        std::make_shared<op::GenericOp>(
            forward_node->get_name() + "_exp_deri", "PowerBackwardExponent", op::OpConfig::any()),
        {x, y});
    auto y_grad = graph->add_node_and_edge(std::make_shared<op::Multiply>(),
                                           {outputs_grad[0], GNodeIndex{exp_deri, 0}});
    return GNodeIndexVector{GNodeIndex{x_grad}, GNodeIndex{y_grad}};
});

// REGISTER_BACKWARD_TRANSLATOR(Power).translator([](std::shared_ptr<GNode> forward_node,
//                                                   const GNodeIndexVector& outputs_grad,
//                                                   std::shared_ptr<nnfusion::graph::Graph> graph)
//                                                    -> GNodeIndexVector {
//     NNFUSION_CHECK(outputs_grad.size() == 1) << "power have only 1 output, but "
//                                              << outputs_grad.size() << " outputs_grad provided";
//     // z = x**y, x_grad = z_grad * y * x ** (y - 1) = z_grad * y * z / x, y_grad = z_grad * x ** y * lnx = z_grad * z * lnx
//     // update: cannot computing x_grad by z_grad * y * z / x, because it returns nan for where x=0, it's the same for y_grad
//     // we should implement a dedicate power_backward op
//     NNFUSION_LOG(NNFUSION_WARNING) << "Current power backward might lead nan";
//     auto x = get_node_input(forward_node, 0);
//     auto y = get_node_input(forward_node, 1);
//     auto z = get_node_output(forward_node, 0);
//     auto z_grad_mul_z =
//         graph->add_node_and_edge(std::make_shared<op::Multiply>(), {outputs_grad[0], z});

//     auto one_op =
//         std::make_shared<op::Constant>(element::f32, nnfusion::Shape{}, std::vector<float>{1});
//     auto one = GNodeIndex{graph->add_node_and_edge(one_op, GNodeIndexVector{}), 0};
//     std::tie(y, one) = numpy_broadcast(std::make_pair(y, one), graph);
//     auto y_sub_one = graph->add_node_and_edge(std::make_shared<op::Subtract>(), {y, one});
//     auto x_power =
//         graph->add_node_and_edge(std::make_shared<op::Power>(), {x, GNodeIndex{y_sub_one, 0}});
//     auto y_mul_x_power =
//         graph->add_node_and_edge(std::make_shared<op::Multiply>(), {y, GNodeIndex{x_power, 0}});
//     auto x_grad = graph->add_node_and_edge(std::make_shared<op::Multiply>(),
//                                            {outputs_grad[0], GNodeIndex{y_mul_x_power, 0}});

//     auto lnx = graph->add_node_and_edge(std::make_shared<op::Log>(), {x});
//     auto y_grad = graph->add_node_and_edge(std::make_shared<op::Multiply>(), {z_grad_mul_z, lnx});
//     return GNodeIndexVector{GNodeIndex{x_grad}, GNodeIndex{y_grad}};
// });