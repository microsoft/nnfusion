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

REGISTER_BACKWARD_TRANSLATOR(MatMulAdd).translator(
    [](std::shared_ptr<GNode> forward_node,
       const GNodeIndexVector& outputs_grad,
       std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        // Y = AB + C, A_grad = Y_grad.B^T, B_grad = A^T.Y_grad, C_grad = Y_grad
        NNFUSION_CHECK(outputs_grad.size() == 1) << "MatMulAdd have only 1 output, but "
                                                 << outputs_grad.size() << " outputs_grad provided";
        auto generic_op = std::dynamic_pointer_cast<op::GenericOp>(forward_node->get_op_ptr());
        auto& cfg = generic_op->localOpConfig.getRoot();
        bool trans_A = cfg["trans_A"];
        bool trans_B = cfg["trans_B"];

        auto A = get_node_input(forward_node, 0);
        auto B = get_node_input(forward_node, 1);
        auto C = get_node_input(forward_node, 2);

        if (trans_A && trans_B)
        {
            auto A_grad_op = std::make_shared<nnfusion::op::Dot>(0, false, true, true);
            A_grad_op->set_name(forward_node->get_name() + "_a_grad");
            auto A_grad_node = graph->add_node_and_edge(A_grad_op, {B, outputs_grad[0]});

            auto B_grad_op = std::make_shared<nnfusion::op::Dot>(0, false, true, true);
            B_grad_op->set_name(forward_node->get_name() + "_b_grad");
            auto B_grad_node = graph->add_node_and_edge(B_grad_op, {outputs_grad[0], A});
            return GNodeIndexVector{
                GNodeIndex{A_grad_node, 0}, GNodeIndex{B_grad_node, 0}, outputs_grad[0]};
        }
        else if (trans_A)
        {
            auto A_grad_op = std::make_shared<nnfusion::op::Dot>(0, false, false, true);
            A_grad_op->set_name(forward_node->get_name() + "_a_grad");
            auto A_grad_node = graph->add_node_and_edge(A_grad_op, {B, outputs_grad[0]});

            auto B_grad_op = std::make_shared<nnfusion::op::Dot>(0, false, false, false);
            B_grad_op->set_name(forward_node->get_name() + "_b_grad");
            auto B_grad_node = graph->add_node_and_edge(B_grad_op, {A, outputs_grad[0]});
            return GNodeIndexVector{
                GNodeIndex{A_grad_node, 0}, GNodeIndex{B_grad_node, 0}, outputs_grad[0]};
        }
        else if (trans_B)
        {
            auto A_grad_op = std::make_shared<nnfusion::op::Dot>(0, false, false, false);
            A_grad_op->set_name(forward_node->get_name() + "_a_grad");
            auto A_grad_node = graph->add_node_and_edge(A_grad_op, {outputs_grad[0], B});

            auto B_grad_op = std::make_shared<nnfusion::op::Dot>(0, false, true, false);
            B_grad_op->set_name(forward_node->get_name() + "_b_grad");
            auto B_grad_node = graph->add_node_and_edge(B_grad_op, {outputs_grad[0], A});
            return GNodeIndexVector{
                GNodeIndex{A_grad_node, 0}, GNodeIndex{B_grad_node, 0}, outputs_grad[0]};
        }
        else
        {
            auto A_grad_op = std::make_shared<nnfusion::op::Dot>(0, false, false, true);
            A_grad_op->set_name(forward_node->get_name() + "_a_grad");
            auto A_grad_node = graph->add_node_and_edge(A_grad_op, {outputs_grad[0], B});

            auto B_grad_op = std::make_shared<nnfusion::op::Dot>(0, false, true, false);
            B_grad_op->set_name(forward_node->get_name() + "_b_grad");
            auto B_grad_node = graph->add_node_and_edge(B_grad_op, {A, outputs_grad[0]});
            return GNodeIndexVector{
                GNodeIndex{A_grad_node, 0}, GNodeIndex{B_grad_node, 0}, outputs_grad[0]};
        }
    });