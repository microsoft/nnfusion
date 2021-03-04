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

REGISTER_BACKWARD_TRANSLATOR(Parameter).translator([](std::shared_ptr<GNode> forward_node,
                                                      const GNodeIndexVector& outputs_grad,
                                                      std::shared_ptr<nnfusion::graph::Graph> graph)
                                                       -> GNodeIndexVector {
    NNFUSION_CHECK(outputs_grad.size() == 1) << "parameter have only 1 output, but "
                                             << outputs_grad.size() << " outputs_grad provided";
    auto graph_outputs = graph->get_outputs();
    auto parameter_op = std::dynamic_pointer_cast<op::Parameter>(forward_node->get_op_ptr());
    ///\todo support other optimizer, support scheduled learning rate
    if (parameter_op->require_grad())
    {
        std::unordered_set<std::shared_ptr<GNode>> param_consumers;
        for (const auto& consumer_edge : forward_node->get_out_edges())
        {
            param_consumers.insert(consumer_edge->get_dst());
        }

        nnfusion::op::OpConfig::any myConfig;
        myConfig["learning_rate"] =
            nnfusion::pass::graph::autodiff::training_optimizer_configs["learning_rate"];
        auto opt_op = std::make_shared<nnfusion::op::GenericOp>(
            forward_node->get_name() + "_sgd", "ApplyGradient", myConfig);
        auto opt_node =
            graph->add_node_and_edge(opt_op, {get_node_output(forward_node, 0), outputs_grad[0]});

        for (auto consumer : param_consumers)
        {
            graph->add_edge(consumer, -1, opt_node, -1);
        }
        graph_outputs.push_back(opt_node);
    }
    else
    {
        graph_outputs.push_back(outputs_grad[0].gnode);
    }

    graph->set_outputs(graph_outputs);
    return GNodeIndexVector{};
});