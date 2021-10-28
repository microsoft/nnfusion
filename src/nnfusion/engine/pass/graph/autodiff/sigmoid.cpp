// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Sigmoid).translator(
    [](std::shared_ptr<GNode> forward_node,
       const GNodeIndexVector& outputs_grad,
       std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        NNFUSION_CHECK(outputs_grad.size() == 1) << "sigmoid have only 1 output, but "
                                                 << outputs_grad.size() << " outputs_grad provided";
        auto x_grad = graph->add_node_and_edge(std::make_shared<op::SigmoidBackprop>(),
                                               {get_node_input(forward_node, 0), outputs_grad[0]});
        return GNodeIndexVector{GNodeIndex{x_grad, 0}};
    });
