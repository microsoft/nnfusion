// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Negative).translator(
    [](std::shared_ptr<GNode> forward_node,
       const GNodeIndexVector& outputs_grad,
       std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        NNFUSION_CHECK(outputs_grad.size() == 1) << "negative have only 1 output, but "
                                                 << outputs_grad.size() << " outputs_grad provided";
        // f = negative(x), x_grad = -f_grad
        auto x_grad_op = std::make_shared<op::Negative>();
        x_grad_op->set_name(forward_node->get_name() + "_x_grad");
        auto x_grad = graph->add_node_and_edge(x_grad_op, {outputs_grad[0]});

        return GNodeIndexVector{GNodeIndex{x_grad, 0}};
    });