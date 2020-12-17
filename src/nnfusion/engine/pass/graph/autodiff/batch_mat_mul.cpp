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

REGISTER_BACKWARD_TRANSLATOR(BatchMatMul)
    .translator([](std::shared_ptr<GNode> forward_node,
                   const GNodeIndexVector& outputs_grad,
                   std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        // Y = AB, A_grad = Y_grad.B^T, B_grad = A^T.Y_grad
        NNFUSION_CHECK(outputs_grad.size() == 1) << "batch_mat_mul have only 1 output, but "
                                                 << outputs_grad.size() << " outputs_grad provided";
        auto batchmatmul_op =
            std::dynamic_pointer_cast<nnfusion::op::GenericOp>(forward_node->get_op_ptr());
        bool trans_A = batchmatmul_op->localOpConfig.getRoot()["adj_x"]["b"];
        bool trans_B = batchmatmul_op->localOpConfig.getRoot()["adj_y"]["b"];

        auto A = get_node_input(forward_node, 0);
        auto B = get_node_input(forward_node, 1);

        if (trans_A && trans_B)
        {
            nnfusion::op::OpConfig::any myConfigA;
            myConfigA["adj_x"]["b"] = true;
            myConfigA["adj_y"]["b"] = true;
            auto A_grad_op = std::make_shared<nnfusion::op::GenericOp>(
                forward_node->get_name() + "_a_grad", "BatchMatMul", myConfigA);
            auto A_grad_node = graph->add_node_and_edge(A_grad_op, {B, outputs_grad[0]});

            nnfusion::op::OpConfig::any myConfigB;
            myConfigB["adj_x"]["b"] = true;
            myConfigB["adj_y"]["b"] = true;
            auto B_grad_op = std::make_shared<nnfusion::op::GenericOp>(
                forward_node->get_name() + "_b_grad", "BatchMatMul", myConfigB);
            auto B_grad_node = graph->add_node_and_edge(B_grad_op, {outputs_grad[0], A});
            return GNodeIndexVector{GNodeIndex{A_grad_node, 0}, GNodeIndex{B_grad_node, 0}};
        }
        else if (trans_A)
        {
            nnfusion::op::OpConfig::any myConfigA;
            myConfigA["adj_x"]["b"] = false;
            myConfigA["adj_y"]["b"] = true;
            auto A_grad_op = std::make_shared<nnfusion::op::GenericOp>(
                forward_node->get_name() + "_a_grad", "BatchMatMul", myConfigA);
            auto A_grad_node = graph->add_node_and_edge(A_grad_op, {B, outputs_grad[0]});

            nnfusion::op::OpConfig::any myConfigB;
            myConfigB["adj_x"]["b"] = false;
            myConfigB["adj_y"]["b"] = false;
            auto B_grad_op = std::make_shared<nnfusion::op::GenericOp>(
                forward_node->get_name() + "_b_grad", "BatchMatMul", myConfigB);
            auto B_grad_node = graph->add_node_and_edge(B_grad_op, {A, outputs_grad[0]});
            return GNodeIndexVector{GNodeIndex{A_grad_node, 0}, GNodeIndex{B_grad_node, 0}};
        }
        else if (trans_B)
        {
            nnfusion::op::OpConfig::any myConfigA;
            myConfigA["adj_x"]["b"] = false;
            myConfigA["adj_y"]["b"] = false;
            auto A_grad_op = std::make_shared<nnfusion::op::GenericOp>(
                forward_node->get_name() + "_a_grad", "BatchMatMul", myConfigA);
            auto A_grad_node = graph->add_node_and_edge(A_grad_op, {outputs_grad[0], B});

            nnfusion::op::OpConfig::any myConfigB;
            myConfigB["adj_x"]["b"] = true;
            myConfigB["adj_y"]["b"] = false;
            auto B_grad_op = std::make_shared<nnfusion::op::GenericOp>(
                forward_node->get_name() + "_b_grad", "BatchMatMul", myConfigB);
            auto B_grad_node = graph->add_node_and_edge(B_grad_op, {outputs_grad[0], A});
            return GNodeIndexVector{GNodeIndex{A_grad_node, 0}, GNodeIndex{B_grad_node, 0}};
        }
        else
        {
            nnfusion::op::OpConfig::any myConfigA;
            myConfigA["adj_x"]["b"] = false;
            myConfigA["adj_y"]["b"] = true;
            auto A_grad_op = std::make_shared<nnfusion::op::GenericOp>(
                forward_node->get_name() + "_a_grad", "BatchMatMul", myConfigA);
            auto A_grad_node = graph->add_node_and_edge(A_grad_op, {outputs_grad[0], B});

            nnfusion::op::OpConfig::any myConfigB;
            myConfigB["adj_x"]["b"] = true;
            myConfigB["adj_y"]["b"] = false;
            auto B_grad_op = std::make_shared<nnfusion::op::GenericOp>(
                forward_node->get_name() + "_b_grad", "BatchMatMul", myConfigB);
            auto B_grad_node = graph->add_node_and_edge(B_grad_op, {A, outputs_grad[0]});
            return GNodeIndexVector{GNodeIndex{A_grad_node, 0}, GNodeIndex{B_grad_node, 0}};
        }
    });