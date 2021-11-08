// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Abs).translator(
    [](std::shared_ptr<GNode> forward_node,
       const GNodeIndexVector& outputs_grad,
       std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        NNFUSION_CHECK(outputs_grad.size() == 1) << "abs have only 1 output, but "
                                                 << outputs_grad.size() << " outputs_grad provided";
        // f = abs(x), x_grad = f_grad * sign(x)
        auto x = get_node_input(forward_node, 0);
        auto x_sign_op = std::make_shared<op::Sign>();
        x_sign_op->set_name(forward_node->get_name() + "_x_sign");
        auto x_sign = graph->add_node_and_edge(x_sign_op, {x});
        auto x_grad_op = std::make_shared<op::Multiply>();
        x_grad_op->set_name(forward_node->get_name() + "_x_grad");
        auto x_grad = graph->add_node_and_edge(x_grad_op, {outputs_grad[0], GNodeIndex{x_sign}});

        return GNodeIndexVector{GNodeIndex{x_grad, 0}};
    });