// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Roll).translator(
    [](std::shared_ptr<GNode> forward_node,
       const GNodeIndexVector& outputs_grad,
       std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        NNFUSION_CHECK(outputs_grad.size() == 1) << "Roll have only 1 output, but "
                                                 << outputs_grad.size() << " outputs_grad provided";
        NNFUSION_CHECK_NOT_NULLPTR(forward_node);
        auto generic_op = static_pointer_cast<nnfusion::op::GenericOp>(forward_node->get_op_ptr());
        std::vector<int> shifts = generic_op->localOpConfig.getRoot()["shifts"];
        std::vector<size_t> dims = generic_op->localOpConfig.getRoot()["dims"];

        auto f_shifts = std::vector<int>(shifts.size());
        std::transform(
            shifts.rbegin(), shifts.rend(), f_shifts.begin(), [](int i) -> int { return -i; });
        std::reverse(dims.begin(), dims.end());

        op::OpConfig::any myConfig;
        myConfig["shifts"] = f_shifts;
        myConfig["dims"] = dims;

        auto x_grad_op = std::make_shared<nnfusion::op::GenericOp>(
            forward_node->get_name() + "_grad", "Roll", myConfig);
        auto x_grad = graph->add_node_and_edge(x_grad_op, {outputs_grad[0]});

        return GNodeIndexVector{GNodeIndex{x_grad, 0}};
    });
