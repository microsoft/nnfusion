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

REGISTER_BACKWARD_TRANSLATOR(GatherV2).translator(
    [](std::shared_ptr<GNode> forward_node,
       const GNodeIndexVector& outputs_grad,
       std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        NNFUSION_CHECK(outputs_grad.size() == 1) << "gather_v2 have only 1 output, but "
                                                 << outputs_grad.size() << " outputs_grad provided";
        auto input_shape = get_node_input(forward_node, 0).get_shape();
        auto indices_index = get_node_input(forward_node, 1);
        auto gather_v2_index = get_node_output(forward_node, 0);
        auto gather_v2_op =
            std::dynamic_pointer_cast<nnfusion::op::GenericOp>(forward_node->get_op_ptr());

        int axis = gather_v2_op->localOpConfig.getRoot()["axis"];
        std::vector<int> x_shape;
        for (int d : input_shape)
            x_shape.push_back(d);

        nnfusion::op::OpConfig::any zerosConfig;
        zerosConfig["shape"] = x_shape;
        auto zeros_op = std::make_shared<nnfusion::op::GenericOp>(
            forward_node->get_name() + "_zeros", "Zeros", zerosConfig);
        auto zeros_gnode = graph->add_node_and_edge(zeros_op, GNodeIndexVector{});

        nnfusion::op::OpConfig::any myConfig;
        myConfig["axis"] = axis;
        myConfig["x_shape"] = x_shape;
        auto gather_grad_op = std::make_shared<nnfusion::op::GenericOp>(
            forward_node->get_name() + "_x_grad", "GatherGrad", myConfig);
        auto x_grad = graph->add_node_and_edge(
            gather_grad_op, {indices_index, outputs_grad[0], GNodeIndex{zeros_gnode}});

        return GNodeIndexVector{GNodeIndex{x_grad, 0},
                                nnfusion::pass::graph::autodiff::DiffEngine::EMPTY_GNODE_INDEX};
    });