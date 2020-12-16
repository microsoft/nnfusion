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

REGISTER_BACKWARD_TRANSLATOR(CrossEntropyAvgLossWithLabels)
    .translator([](std::shared_ptr<GNode> forward_node,
                   const GNodeIndexVector& outputs_grad,
                   std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        NNFUSION_CHECK(outputs_grad.size() == 1)
            << "CrossEntropyAvgLossWithLabels have only 1 output, but " << outputs_grad.size()
            << " outputs_grad provided";
        auto logit = get_node_input(forward_node, 0);
        auto label = get_node_input(forward_node, 1);

        auto ce_op = std::dynamic_pointer_cast<op::GenericOp>(forward_node->get_op_ptr());
        auto& ce_cfg = ce_op->localOpConfig.getRoot();
        bool in_log_space = ce_cfg["in_log_space"];

        nnfusion::op::OpConfig::any onehot_config;
        onehot_config["axis"] = -1;
        onehot_config["depth"] = logit.get_shape()[1];
        onehot_config["off_value"] = 0;
        onehot_config["on_value"] = 1;
        onehot_config["T"] = "float";

        auto onehot_op = std::make_shared<nnfusion::op::GenericOp>(
            label.gnode->get_name() + "_onehot", "OneHot", onehot_config);
        auto onehot_node = graph->add_node_and_edge(onehot_op, {label});

        nnfusion::AxisSet broadcast_axes{1};
        auto outputs_grad_broadcasted_op =
            std::make_shared<op::Broadcast>(logit.get_shape(), broadcast_axes);
        outputs_grad_broadcasted_op->set_name(forward_node->get_name() + "_outputs_grad_broadcast");
        auto outputs_grad_broadcasted_node =
            graph->add_node_and_edge(outputs_grad_broadcasted_op, {outputs_grad[0]});

        if (in_log_space)
        {
            auto label_onehot_div_logit_op = std::make_shared<op::DivNoNan>();
            label_onehot_div_logit_op->set_name(forward_node->get_name() +
                                                "_label_onehot_divnonan_logit");
            auto label_onehot_div_logit_node = graph->add_node_and_edge(
                label_onehot_div_logit_op, {GNodeIndex{onehot_node}, logit});

            auto neg_label_onehot_div_logit_op = std::make_shared<op::Negative>();
            neg_label_onehot_div_logit_op->set_name(forward_node->get_name() +
                                                    "_neg_label_onehot_div_logit");
            auto neg_label_onehot_div_logit_node = graph->add_node_and_edge(
                neg_label_onehot_div_logit_op, {GNodeIndex{label_onehot_div_logit_node}});

            auto logit_grad_op = std::make_shared<op::Multiply>();
            logit_grad_op->set_name(forward_node->get_name() + "_logit_grad");
            auto logit_grad_node =
                graph->add_node_and_edge(logit_grad_op,
                                         {GNodeIndex{outputs_grad_broadcasted_node},
                                          GNodeIndex{neg_label_onehot_div_logit_node}});

            return GNodeIndexVector{GNodeIndex{logit_grad_node, 0},
                                    nnfusion::pass::graph::autodiff::DiffEngine::EMPTY_GNODE_INDEX};
        }
        else
        {
            auto neg_label_onehot_op = std::make_shared<op::Negative>();
            neg_label_onehot_op->set_name(forward_node->get_name() + "_neg_label_onehot");
            auto neg_label_onehot_node =
                graph->add_node_and_edge(neg_label_onehot_op, {GNodeIndex{onehot_node}});

            auto logit_grad_op = std::make_shared<op::Multiply>();
            logit_grad_op->set_name(forward_node->get_name() + "_logit_grad");
            auto logit_grad_node = graph->add_node_and_edge(
                logit_grad_op,
                {GNodeIndex{outputs_grad_broadcasted_node}, GNodeIndex{neg_label_onehot_node}});

            return GNodeIndexVector{GNodeIndex{logit_grad_node, 0},
                                    nnfusion::pass::graph::autodiff::DiffEngine::EMPTY_GNODE_INDEX};
        }

    });