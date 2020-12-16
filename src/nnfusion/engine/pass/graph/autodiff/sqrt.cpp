// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Sqrt).translator(
    [](std::shared_ptr<GNode> forward_node,
       const GNodeIndexVector& outputs_grad,
       std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        NNFUSION_CHECK(outputs_grad.size() == 1) << "power have only 1 output, but "
                                                 << outputs_grad.size() << " outputs_grad provided";
        // y = x**(1/2), x_grad = y_grad * (1/2) * x**(-1/2) = y_grad / (y + y)
        auto x = get_node_input(forward_node, 0);
        auto y = get_node_output(forward_node, 0);
        auto y_mul_2 = graph->add_node_and_edge(std::make_shared<op::Add>(), {y, y});
        auto x_grad = graph->add_node_and_edge(std::make_shared<op::Divide>(),
                                               {outputs_grad[0], GNodeIndex(y_mul_2)});
        return GNodeIndexVector{GNodeIndex{x_grad}};
    });