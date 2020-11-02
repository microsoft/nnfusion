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

REGISTER_BACKWARD_TRANSLATOR(LayerNorm).translator([](std::shared_ptr<GNode> forward_node,
                                                      const GNodeIndexVector& outputs_grad,
                                                      std::shared_ptr<nnfusion::graph::Graph> graph)
                                                       -> GNodeIndexVector {
    // y, x_mean, x_inv_std_var = layer_norm(x, weight, bias), N is batch size, D is feature size
    //
    // bias_grad = sum(y_grad, N_dims)
    //
    // xmu = x - x_mean
    // weight_grad = sum(y_grad * xmu * x_inv_std_var, N_dims)
    //
    // part1 = y_grad * weight * x_inv_std_var
    // part2 = -0.5 * sum(weight * xmu * x_inv_std_var^3, y_grad, D_dims) / D
    // part3 = -1 * sum(part1) / D
    // x_grad = part1 + part2 + part3

    auto y_grad_index = outputs_grad.at(0);                    // 2, 128, 1024
    auto x_index = get_node_input(forward_node, 0);            // 2, 128, 1024
    auto weight_index = get_node_input(forward_node, 1);       // 1024
    auto mean_index = get_node_output(forward_node, 1);        // 2, 128, 1
    auto inv_std_var_index = get_node_output(forward_node, 2); // 2, 128, 1

    auto input_shape = x_index.get_shape();
    auto input_rank = input_shape.size();

    auto layer_norm_op = std::dynamic_pointer_cast<op::GenericOp>(forward_node->get_op_ptr());

    auto& cfg = layer_norm_op->localOpConfig.getRoot();
    float eps = cfg["epsilon"];
    eps = eps > 0 ? eps : 1e-5f;
    int axis = cfg["axis"];
    axis += axis < 0 ? input_rank : 0;

    Shape normalized_shape(std::next(input_shape.begin(), axis), input_shape.end());
    auto normalized_rank = normalized_shape.size();
    auto batch_rank = input_rank - normalized_rank;

    auto num_feature = nnfusion::shape_size(normalized_shape);

    std::vector<size_t> reduction_axes(batch_rank);
    std::iota(reduction_axes.begin(), reduction_axes.end(), 0);
    auto bias_grad_op = std::make_shared<op::Sum>(reduction_axes);

    std::vector<size_t> norm_reduction_axes(normalized_rank);
    std::iota(norm_reduction_axes.begin(), norm_reduction_axes.end(), input_rank - normalized_rank);

    auto bias_grad_gnode = graph->add_node_and_edge(bias_grad_op, {y_grad_index}); // 1024

    // xmu = x - mean
    std::tie(x_index, mean_index) =
        graph::numpy_broadcast(std::make_pair(x_index, mean_index), graph);
    auto xmu_gnode = graph->add_node_and_edge(std::make_shared<op::Subtract>(),
                                              {x_index, mean_index}); // 2, 128, 1024
    auto xmu_index = GNodeIndex{xmu_gnode};

    // weight_grad
    std::tie(xmu_index, inv_std_var_index) =
        numpy_broadcast(std::make_pair(xmu_index, inv_std_var_index), graph);
    auto weight_grad_gnode_temp1 =
        graph->add_node_and_edge(std::make_shared<op::Multiply>(), {xmu_index, inv_std_var_index});
    auto weight_grad_gnode_temp2 = graph->add_node_and_edge(
        std::make_shared<op::Multiply>(),
        {GNodeIndex{weight_grad_gnode_temp1}, y_grad_index}); // 2, 128, 1024
    auto weight_grad_op = std::make_shared<op::Sum>(reduction_axes);
    auto weight_grad_gnode =
        graph->add_node_and_edge(weight_grad_op, {weight_grad_gnode_temp2}); // 1024

    // x_grad
    // first part
    std::tie(y_grad_index, weight_index) =
        numpy_broadcast(std::make_pair(y_grad_index, weight_index), graph);
    auto xhat_grad_gnode = graph->add_node_and_edge(std::make_shared<op::Multiply>(),
                                                    {y_grad_index, weight_index}); // 2, 128, 1024
    auto xhat_x_grad_index = inv_std_var_index;
    auto first_part_gnode = graph->add_node_and_edge(
        std::make_shared<op::Multiply>(), {GNodeIndex{xhat_grad_gnode}, xhat_x_grad_index});

    // second part
    auto three_op = std::make_shared<op::Constant>(
        nnfusion::element::f32, inv_std_var_index.get_shape(), std::vector<float>{3.0});
    auto three_gnode = graph->add_node_and_edge(three_op, GNodeVector({}));
    auto inv_var_pow_gnode = graph->add_node_and_edge(std::make_shared<op::Power>(),
                                                      {inv_std_var_index, GNodeIndex{three_gnode}});
    auto second_part_gnode_temp = graph->add_node_and_edge(
        std::make_shared<op::Multiply>(), {GNodeIndex{xhat_grad_gnode}, xmu_index});
    second_part_gnode_temp = graph->add_node_and_edge(std::make_shared<op::Multiply>(),
                                                      {second_part_gnode_temp, inv_var_pow_gnode});
    second_part_gnode_temp = graph->add_node_and_edge(
        std::make_shared<op::Sum>(norm_reduction_axes), {second_part_gnode_temp}); // 2, 128
    auto shape_with_keep = second_part_gnode_temp->get_shape();
    shape_with_keep.push_back(1);
    nnfusion::AxisVector ng_axis_order(second_part_gnode_temp->get_shape().size());
    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
    second_part_gnode_temp =
        graph->add_node_and_edge(std::make_shared<op::Reshape>(ng_axis_order, shape_with_keep),
                                 {second_part_gnode_temp}); // 2, 128, 1
    std::tie(xmu_gnode, second_part_gnode_temp) =
        numpy_broadcast(std::make_pair(xmu_gnode, second_part_gnode_temp), graph);
    second_part_gnode_temp = graph->add_node_and_edge(
        std::make_shared<op::Multiply>(), {second_part_gnode_temp, xmu_gnode}); // 2, 128, 1024
    auto divisor_op = std::make_shared<op::Constant>(
        nnfusion::element::f32, input_shape, std::vector<size_t>{num_feature});
    auto divisor_gnode = graph->add_node_and_edge(divisor_op, GNodeVector({}));
    auto second_part_gnode = graph->add_node_and_edge(std::make_shared<op::Divide>(),
                                                      {second_part_gnode_temp, divisor_gnode});

    // third part
    auto third_part_gnode_temp = graph->add_node_and_edge(
        std::make_shared<op::Sum>(norm_reduction_axes), {first_part_gnode}); // 2, 128
    third_part_gnode_temp =
        graph->add_node_and_edge(std::make_shared<op::Reshape>(ng_axis_order, shape_with_keep),
                                 {third_part_gnode_temp}); // 2, 128, 1
    std::tie(third_part_gnode_temp, divisor_gnode) =
        numpy_broadcast(std::make_pair(third_part_gnode_temp, divisor_gnode), graph);
    auto third_part_gnode = graph->add_node_and_edge(
        std::make_shared<op::Divide>(), {third_part_gnode_temp, divisor_gnode}); // 2, 128, 1024

    auto x_grad_temp = graph->add_node_and_edge(std::make_shared<op::Subtract>(),
                                                {first_part_gnode, second_part_gnode});
    auto x_grad_op = std::make_shared<op::Subtract>();
    auto x_grad = graph->add_node_and_edge(x_grad_op, {x_grad_temp, third_part_gnode});

    return {
        GNodeIndex{x_grad, 0}, GNodeIndex{weight_grad_gnode, 0}, GNodeIndex{bias_grad_gnode, 0}};
});