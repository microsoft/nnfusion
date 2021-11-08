#include "backward_registry.hpp"
#include "nnfusion/core/graph/util/autobroadcast.hpp"
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

REGISTER_BACKWARD_TRANSLATOR(AvgPool).translator(
    [](std::shared_ptr<GNode> forward_node,
       const GNodeIndexVector& outputs_grad,
       std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        auto x = get_node_input(forward_node, 0);
        auto y = get_node_output(forward_node, 0);
        auto y_grad = outputs_grad.at(0);

        auto avgpool = std::dynamic_pointer_cast<op::AvgPool>(forward_node->get_op_ptr());

        auto forward_arg_shape = x.get_shape();
        auto window_shape = avgpool->get_window_shape();
        auto window_movement_strides = avgpool->get_window_movement_strides();
        auto padding_below = avgpool->get_padding_below();
        auto padding_above = avgpool->get_padding_above();
        auto include_padding_in_avg_computation = avgpool->get_include_padding_in_avg_computation();

        auto avgpoolbackprop =
            std::make_shared<op::AvgPoolBackprop>(forward_arg_shape,
                                                  window_shape,
                                                  window_movement_strides,
                                                  padding_below,
                                                  padding_above,
                                                  include_padding_in_avg_computation);

        auto x_grad = graph->add_node_and_edge(avgpoolbackprop, {x, y, y_grad});

        return GNodeIndexVector{GNodeIndex{x_grad, 0}};
    });