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

// Microsoft (c) 2020, NNFusion Team

#include <climits>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "autodiff/backward_registry.hpp"
#include "autodiff_pass.hpp"

DEFINE_bool(fautodiff, false, "Add backward graph.");
DEFINE_string(ftraining_optimizer,
              "{}",
              "Configs for training optimizer (expressed in json string).");

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::pass::graph::autodiff;

bool AutodiffPass::run_on_graph(
    std::shared_ptr<Graph>& graph,
    std::shared_ptr<vector<vector<float>>> backward_inputs = nullptr)
{
    bool enable_autodiff = FLAGS_fautodiff;
    if (!enable_autodiff)
        return true;

    nnfusion::pass::graph::autodiff::training_optimizer_configs =
        nlohmann::json::parse(FLAGS_ftraining_optimizer);
    {
        // process training_optimizer_configs
        // TODO: support other optimizers
        NNFUSION_CHECK(training_optimizer_configs.find("optimizer") !=
                       training_optimizer_configs.end())
            << "Training optimizer should be set in -ftraining_optimizer.";
        NNFUSION_CHECK(training_optimizer_configs["optimizer"] == "SGD")
            << "NNFusion only support SGD optimizer yet.";
        NNFUSION_CHECK(training_optimizer_configs.find("learning_rate") !=
                       training_optimizer_configs.end())
            << "Cannot find learning_rate in training_optimizer.";
    }

    // assume graph outputs are loss
    GNodeIndexVector outputs_index;
    GNodeIndexVector outputs_grad;
    for (size_t i = 0; i < graph->get_output_size(); i++)
    {
        auto gnode = graph->get_outputs()[i];
        NNFUSION_CHECK(gnode->get_output_size() == 1);
        outputs_index.emplace_back(gnode, 0);
        std::shared_ptr<op::Constant> one_op;
        if (backward_inputs != nullptr && i < backward_inputs->size()) {
            one_op = std::make_shared<op::Constant>(element::f32, gnode->get_shape(), backward_inputs->at(i));
        } else {
            one_op = std::make_shared<op::Constant>(element::f32, gnode->get_shape(), std::vector<float>{1});
        }
        one_op->set_name(gnode->get_name() + "_grad");
        auto one = graph->add_node_and_edge(one_op, GNodeVector());
        outputs_grad.emplace_back(one, 0);
    }

    DiffEngine(graph).differentiate_graph(outputs_index, outputs_grad);

    return true;
}
