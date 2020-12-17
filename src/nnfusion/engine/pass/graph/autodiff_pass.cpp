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

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::pass::graph::autodiff;

bool AutodiffPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    bool enable_autodiff = FLAGS_fautodiff;
    if (!enable_autodiff)
        return true;

    // assume graph outputs are loss
    GNodeIndexVector outputs_index;
    GNodeIndexVector outputs_grad;
    for (auto gnode : graph->get_outputs())
    {
        NNFUSION_CHECK(gnode->get_output_size() == 1);
        outputs_index.emplace_back(gnode, 0);
        auto one_op =
            std::make_shared<op::Constant>(element::f32, gnode->get_shape(), std::vector<float>{1});
        one_op->set_name("out_grad");
        auto one = graph->add_node_and_edge(one_op, GNodeVector());
        outputs_grad.emplace_back(one, 0);
    }

    DiffEngine(graph).differentiate_graph(outputs_index, outputs_grad);

    return true;
}
